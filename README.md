# BRB Takin a Trip to the MOOOOON🚀
before developing run the following in command line: 
~~~
python3 -m venv venv
pip install -r requirements.txt
~~~

## Trade assets, namely crypto🌟
- Using this module, you can create LSTM model for a variety of securities. You can then use these to deploy and put what they learn to work!

## Quick start (modern)
- Copy env file: `cp .env.example .env` and fill in `API_KEY`
- Run: `python main.py`

## Tooling
- Lint/format: `ruff check .` and `ruff format .`
- Tests: `pytest`

## Notes
- `CoinMarketData` normalizes CoinMarketCap listings into a DataFrame and extracts `price` from `quote.<convert>.price`.
- The cache layer now supports a `cache_key` so different endpoints/symbols can be cached separately.
