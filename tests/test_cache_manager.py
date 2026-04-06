import time

from handlers.cache_manager import CacheManager


def test_cache_roundtrip(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    cache = CacheManager(cache_file=cache_file, cache_expiry=60)

    assert cache.load_cache() is None
    cache.save_cache({"x": 1})
    assert cache.load_cache() == {"x": 1}


def test_cache_expiry(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    cache = CacheManager(cache_file=cache_file, cache_expiry=1)
    cache.save_cache({"x": 1})
    assert cache.load_cache() == {"x": 1}
    time.sleep(1.1)
    assert cache.load_cache() is None


def test_cache_key_separates_files(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    cache = CacheManager(cache_file=cache_file, cache_expiry=60)
    cache.save_cache({"x": 1}, cache_key="a")
    cache.save_cache({"x": 2}, cache_key="b")
    assert cache.load_cache(cache_key="a") == {"x": 1}
    assert cache.load_cache(cache_key="b") == {"x": 2}
