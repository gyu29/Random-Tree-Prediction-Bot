import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import requests


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")

VERIFIED_KRX_ENDPOINTS = [
    {
        "name": "stock",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo",
    },
    {
        "name": "security",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getSecuritiesPriceInfo",
    },
    {
        "name": "warrant_certificate",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getPreemptiveRightCertificatePriceInfo",
    },
    {
        "name": "warrant_security",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getPreemptiveRightSecuritiesPriceInfo",
    },
    {
        "name": "etf",
        "url": "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getETFPriceInfo",
    },
    {
        "name": "etn",
        "url": "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getETNPriceInfo",
    },
    {
        "name": "elw",
        "url": "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getELWPriceInfo",
    },
    {
        "name": "bond",
        "url": "https://apis.data.go.kr/1160100/service/GetBondSecuritiesInfoService/getBondPriceInfo",
    },
]


def read_env_file_value(target_key):
    if not os.path.exists(ENV_FILE_PATH):
        return None

    with open(ENV_FILE_PATH, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() != target_key:
                continue
            return value.strip().strip('"').strip("'")

    return None


def get_runtime_secret():
    env_secret = os.environ.get("KRX_SERVICE_KEY", "").strip()
    file_secret = (read_env_file_value("KRX_SERVICE_KEY") or "").strip()
    return env_secret or file_secret or None


def mask_secret(secret):
    if not secret:
        return "<missing>"
    if len(secret) <= 10:
        return secret
    return f"{secret[:6]}...{secret[-4:]}"


def parse_response(response):
    text = response.text.strip()

    if not text:
        return {
            "kind": "empty",
            "result_code": None,
            "result_msg": "Empty response",
            "raw_preview": "",
        }

    if response.status_code in {401, 403} or "unauthorized" in text.lower():
        return {
            "kind": "unauthorized",
            "result_code": str(response.status_code),
            "result_msg": text[:200],
            "raw_preview": text[:200],
        }

    if text.startswith("<"):
        try:
            root = ET.fromstring(text)
            header = root.find("./header")
            return {
                "kind": "xml",
                "result_code": header.findtext("resultCode", default="") if header is not None else "",
                "result_msg": header.findtext("resultMsg", default="") if header is not None else "",
                "raw_preview": text[:200],
            }
        except ET.ParseError:
            return {
                "kind": "invalid_xml",
                "result_code": None,
                "result_msg": "XML parse error",
                "raw_preview": text[:200],
            }

    try:
        data = response.json()
    except json.JSONDecodeError:
        return {
            "kind": "text",
            "result_code": None,
            "result_msg": text[:200],
            "raw_preview": text[:200],
        }

    header = data.get("response", {}).get("header", {})
    return {
        "kind": "json",
        "result_code": header.get("resultCode"),
        "result_msg": header.get("resultMsg"),
        "raw_preview": text[:200],
    }


def test_endpoint(service_key, endpoint):
    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "pageNo": 1,
        "numOfRows": 1,
    }

    try:
        response = requests.get(endpoint["url"], params=params, timeout=20)
    except requests.RequestException as exc:
        return {
            "endpoint": endpoint["name"],
            "ok": False,
            "status": "network_error",
            "detail": str(exc),
        }

    parsed = parse_response(response)
    result_code = parsed.get("result_code")
    result_msg = parsed.get("result_msg") or ""

    if result_code == "00":
        return {
            "endpoint": endpoint["name"],
            "ok": True,
            "status": "authorized",
            "detail": result_msg or "NORMAL SERVICE.",
        }

    if parsed["kind"] == "unauthorized":
        return {
            "endpoint": endpoint["name"],
            "ok": False,
            "status": "unauthorized",
            "detail": result_msg,
        }

    return {
        "endpoint": endpoint["name"],
        "ok": False,
        "status": "unexpected_response",
        "detail": f"HTTP {response.status_code}, code={result_code}, msg={result_msg or parsed['raw_preview']}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test whether your data.go.kr KRX service key is accepted by the verified Korean market endpoints."
    )
    parser.add_argument("--key", help="Override KRX_SERVICE_KEY for this run.")
    args = parser.parse_args()

    service_key = (args.key or get_runtime_secret() or "").strip()
    if not service_key:
        print("KRX_SERVICE_KEY was not found.")
        print("Set it in PowerShell with: $env:KRX_SERVICE_KEY='your_key_here'")
        print("Or put it in .env as: KRX_SERVICE_KEY=your_key_here")
        return 1

    print("data.go.kr KRX key tester")
    print("=" * 50)
    print(f"Using key: {mask_secret(service_key)}")
    print()

    results = [test_endpoint(service_key, endpoint) for endpoint in VERIFIED_KRX_ENDPOINTS]
    authorized = [result for result in results if result["ok"]]

    for result in results:
        state = "PASS" if result["ok"] else "FAIL"
        print(f"[{state}] {result['endpoint']}: {result['status']}")
        print(f"      {result['detail']}")

    print()
    if authorized:
        print(f"Key accepted by {len(authorized)}/{len(results)} verified data.go.kr KRX endpoints.")
        return 0

    print("Key was not accepted by any verified data.go.kr KRX endpoint.")
    print("This usually means the key is incorrect, not approved for these datasets yet, or approval has not propagated.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
