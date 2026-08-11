"""Hardened XML parsing for third-party API responses (e.g. KRX/data.go.kr).

Plain xml.etree.ElementTree happily expands internal entities, so a small
malicious/misbehaving payload can blow up into gigabytes in memory (a
"billion laughs" attack) or otherwise hang the process. defusedxml disables
entity expansion and external entity/DTD resolution while keeping the same
ElementTree-style API, so callers don't need to change how they use the result.
"""
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring as _safe_fromstring


def safe_parse_xml(xml_text):
    """Parse an XML string, raising DefusedXmlException on entity-expansion attempts."""
    return _safe_fromstring(xml_text)


__all__ = ["safe_parse_xml", "DefusedXmlException", "ParseError"]
