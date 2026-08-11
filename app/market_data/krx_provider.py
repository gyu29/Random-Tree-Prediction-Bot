"""Korean market data via data.go.kr KRX endpoints."""
import re
from collections import defaultdict

import pandas as pd

from app.config import KRX_ASSET_TYPE_ALIASES, KRX_DEFAULT_LOOKUP_ORDER, KRX_MARKET_ENDPOINTS
from app.market_data.base import MarketDataProvider
from app.security import SecretManager, SecurityValidationError
from app.xml_safety import safe_parse_xml

_ASSET_TYPE_PATTERN = re.compile(r"^(?P<asset>[A-Za-z]+):(?P<value>.+)$")

# data.go.kr doesn't document a hard rate limit for this API family; this is a
# conservative courtesy pace between paginated/fallback requests (the original
# implementation had no pacing here at all, unlike the Alpha Vantage path).
KRX_MIN_REQUEST_INTERVAL_SECONDS = 0.25


class KRXProvider(MarketDataProvider):
    def __init__(self, service_key=None):
        super().__init__(min_request_interval_seconds=KRX_MIN_REQUEST_INTERVAL_SECONDS)
        self._service_key = service_key

    def _resolve_service_key(self):
        # Re-resolve on every call (rather than caching once) so a key updated via the
        # Settings screen takes effect without needing a brand new provider instance.
        service_key = self._service_key or SecretManager.get_runtime_secret("KRX_SERVICE_KEY", prefer_env_file=True)
        if not service_key:
            raise SecurityValidationError(
                "Missing required secret: KRX_SERVICE_KEY. Set it before requesting Korean market data."
            )
        return service_key

    def _parse_krx_symbol(self, symbol):
        normalized_symbol = (symbol or "").strip()
        match = _ASSET_TYPE_PATTERN.match(normalized_symbol)
        if not match:
            return {"raw_symbol": normalized_symbol, "lookup_symbol": normalized_symbol, "asset_type": None}

        alias = match.group("asset").strip().lower()
        asset_type = KRX_ASSET_TYPE_ALIASES.get(alias)
        lookup_symbol = match.group("value").strip()
        if asset_type and lookup_symbol:
            return {"raw_symbol": normalized_symbol, "lookup_symbol": lookup_symbol, "asset_type": asset_type}

        return {"raw_symbol": normalized_symbol, "lookup_symbol": normalized_symbol, "asset_type": None}

    def _endpoint_specs_for_symbol(self, symbol):
        symbol_info = self._parse_krx_symbol(symbol)
        asset_type = symbol_info["asset_type"]
        endpoint_keys = [asset_type] if asset_type else KRX_DEFAULT_LOOKUP_ORDER
        endpoint_specs = [KRX_MARKET_ENDPOINTS[key] for key in endpoint_keys if key in KRX_MARKET_ENDPOINTS]
        verified_specs = [spec for spec in endpoint_specs if spec.get("verified", False)]
        return symbol_info, verified_specs

    def _build_base_params(self, include_date_range=False):
        params = {"resultType": "json", "numOfRows": 1000 if include_date_range else 10, "pageNo": 1}
        if include_date_range:
            end_date = pd.Timestamp.now().normalize()
            start_date = pd.Timestamp(end_date - pd.DateOffset(months=14)).normalize()
            params["beginBasDt"] = start_date.strftime("%Y%m%d")
            params["endBasDt"] = end_date.strftime("%Y%m%d")
        return params

    def _build_lookup_params(self, symbol, include_date_range=False):
        sym = self._parse_krx_symbol(symbol)["lookup_symbol"]
        base_params = self._build_base_params(include_date_range=include_date_range)

        if sym.startswith("KR") and len(sym) >= 10:
            return [{**base_params, "isinCd": sym}, {**base_params, "likeIsinCd": sym}]
        if sym.isdigit():
            return [{**base_params, "likeSrtnCd": sym}]
        return [{**base_params, "itmsNm": sym}, {**base_params, "likeItmsNm": sym}]

    def _parse_xml_response(self, response_text):
        root = safe_parse_xml(response_text)
        header = root.find("./header")
        body = root.find("./body")
        result_code = header.findtext("resultCode", default="") if header is not None else ""
        result_msg = header.findtext("resultMsg", default="") if header is not None else ""
        total_count = body.findtext("totalCount", default="0") if body is not None else "0"
        items = []
        if body is not None:
            items_container = body.find("items")
            if items_container is not None:
                for item in items_container.findall("item"):
                    items.append({child.tag: child.text for child in item})
        return {
            "response": {
                "header": {"resultCode": result_code, "resultMsg": result_msg},
                "body": {"totalCount": total_count, "items": {"item": items}},
            }
        }

    def _filter_items_for_symbol(self, items, symbol):
        normalized_symbol = self._parse_krx_symbol(symbol)["lookup_symbol"]
        if not items:
            return []

        if normalized_symbol.isdigit():
            exact_items = [item for item in items if str(item.get("srtnCd", "")).strip() == normalized_symbol]
            return exact_items or items

        if normalized_symbol.startswith("KR") and len(normalized_symbol) >= 10:
            exact_items = [item for item in items if str(item.get("isinCd", "")).strip() == normalized_symbol]
            return exact_items or items

        exact_name_items = [item for item in items if str(item.get("itmsNm", "")).strip() == normalized_symbol]
        candidate_items = exact_name_items or items

        grouped_items = defaultdict(list)
        for item in candidate_items:
            identity = str(item.get("srtnCd") or item.get("isinCd") or item.get("itmsNm") or "").strip()
            grouped_items[identity].append(item)

        if len(grouped_items) <= 1:
            return candidate_items

        best_identity = max(grouped_items.items(), key=lambda entry: len(entry[1]))[0]
        return grouped_items[best_identity]

    def _normalize_price_frame(self, items):
        df = pd.DataFrame(items)
        if df.empty:
            return None

        df = df.rename(columns={
            "basDt": "date", "clpr": "close", "mkp": "open", "hipr": "high", "lopr": "low", "trqu": "volume",
        })
        if "date" not in df.columns or "close" not in df.columns:
            return None

        for col in ["open", "high", "low"]:
            if col not in df.columns:
                df[col] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date", "close"])
        if df.empty:
            return None

        df = df.set_index("date")
        return self.finalize_ohlcv_frame(df)

    def _parse_response(self, response, security_type):
        if not response.content:
            print(f"  Empty response from {security_type} endpoint")
            return None, False, True

        response_text = response.text[:200]
        lowered_text = response.text.lower()
        if response.status_code in {401, 403} or "unauthorized" in lowered_text:
            print(f"  Response from {security_type}: {response_text}")
            return None, True, True

        if len(response.content) < 500:
            print(f"  Response from {security_type}: {response_text}")

        try:
            if response.text.lstrip().startswith("<"):
                data = self._parse_xml_response(response.text)
            else:
                data = response.json()
        except Exception as parse_err:
            print(f"  Non-JSON/XML response from {security_type}: {parse_err}")
            print(f"  Response text: {response_text}")
            return None, False, True

        result_code = data.get("response", {}).get("header", {}).get("resultCode", "")
        result_msg = data.get("response", {}).get("header", {}).get("resultMsg", "")
        if result_code != "00":
            print(f"  API Error - Code: {result_code}, Message: {result_msg}")
            return None, result_code in {"20", "30", "31", "32"}, True

        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        total_count = data.get("response", {}).get("body", {}).get("totalCount", 0)
        if isinstance(items, dict):
            items = [items]

        return {"items": items or [], "total_count": total_count}, False, False

    def _fetch_market_items(self, symbol, include_date_range=True):
        service_key = self._resolve_service_key()
        symbol_info, endpoint_specs = self._endpoint_specs_for_symbol(symbol)
        if not endpoint_specs:
            requested_asset = symbol_info["asset_type"] or "unknown"
            print(
                f"No verified KRX endpoint is configured yet for asset type `{requested_asset}`. "
                f"Supported verified KR asset types are: stock, etf, etn, elw, security, warrant, bond."
            )
            return None, None
        param_candidates = self._build_lookup_params(symbol, include_date_range=include_date_range)

        if include_date_range:
            end_date_ts = pd.to_datetime(param_candidates[0]["endBasDt"], format="%Y%m%d")
            start_date_ts = pd.to_datetime(param_candidates[0]["beginBasDt"], format="%Y%m%d")
            print(
                f"Fetching KRX data for {symbol_info['lookup_symbol']} from "
                f"{start_date_ts.strftime('%Y-%m-%d')} to {end_date_ts.strftime('%Y-%m-%d')}"
            )
        else:
            print(f"Resolving KRX instrument for {symbol_info['lookup_symbol']}")

        masked = service_key[:6] + "..." if len(service_key) > 8 else service_key
        print(f"Using KRX service key: {masked} (set `KRX_SERVICE_KEY` to change)")

        authorization_failed = False
        for endpoint_spec in endpoint_specs:
            endpoint_items = []
            print(f"Trying KRX {endpoint_spec['label']} endpoint from {endpoint_spec['guide']}...")
            for params in param_candidates:
                page = 1
                while True:
                    request_params = {**params, "pageNo": page, "serviceKey": service_key}
                    response = self._paced_get(endpoint_spec["url"], request_params, timeout=20)
                    parsed_response, is_auth_failure, should_stop = self._parse_response(response, endpoint_spec["label"])
                    authorization_failed = authorization_failed or is_auth_failure
                    if should_stop and parsed_response is None:
                        break

                    items = self._filter_items_for_symbol(parsed_response["items"], symbol)
                    if not items:
                        break

                    endpoint_items.extend(items)
                    total_count = int(parsed_response["total_count"] or 0)
                    if len(endpoint_items) >= total_count:
                        break
                    page += 1

                endpoint_items = self._filter_items_for_symbol(endpoint_items, symbol)
                if endpoint_items:
                    return endpoint_spec, endpoint_items

        if authorization_failed:
            print("KRX authorization failed.")
            print(
                "This usually means the current `KRX_SERVICE_KEY` is not approved for these data.go.kr "
                "financial APIs, is the wrong key, or approval has not propagated yet."
            )
            return None, None

        return None, None

    def fetch_ohlcv(self, symbol, **kwargs):
        try:
            endpoint_spec, items = self._fetch_market_items(symbol, include_date_range=True)
            if not endpoint_spec or not items:
                print(f"No KRX data found for {symbol}")
                return None

            df = self._normalize_price_frame(items)
            if df is None or df.empty:
                print(f"KRX returned data for {symbol}, but it could not be normalized into OHLCV.")
                return None

            print(
                f"Successfully fetched {len(df)} rows of KRX {endpoint_spec['label']} data for {symbol} "
                f"(from {df.index.min().date()} to {df.index.max().date()})"
            )
            return df
        except SecurityValidationError:
            raise
        except Exception as error:
            print(f"Error fetching/parsing KRX data for {symbol}: {error}")
            return None

    def resolve_company_name(self, symbol):
        try:
            _endpoint_spec, items = self._fetch_market_items(symbol, include_date_range=False)
            if items:
                return items[0].get("itmsNm", self._parse_krx_symbol(symbol)["lookup_symbol"])
        except Exception:
            pass
        return self._parse_krx_symbol(symbol)["lookup_symbol"]
