from data.market_data import CoinMarketData


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, *_args, **_kwargs):
        return _FakeResponse(self._payload)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeCache:
    def load_cache(self, cache_key=None):
        return None

    def save_cache(self, data, cache_key=None):
        return None


class _FakeAPIManager:
    def __init__(self, payload):
        self.session = _FakeSession(payload)
        self.cache_manager = _FakeCache()

    def enforce_rate_limits(self):
        return None


def test_dataframe_transform_extracts_price():
    payload = {
        "data": [
            {"symbol": "BTC", "quote": {"USD": {"price": 123.45}}},
            {"symbol": "ETH", "quote": {"USD": {"price": 67.89}}},
        ]
    }
    api = _FakeAPIManager(payload)
    md = CoinMarketData(api)
    assert md.get_data() is not None
    df = md.dataframe_transform()
    assert df is not None
    assert "price" in df.columns
    assert float(df.loc[0, "price"]) == 123.45

