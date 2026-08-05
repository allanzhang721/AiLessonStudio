# Deploy Explain It!

The production entry point is streamlit_app.py. Each visitor supplies a text API
key in the interface; the key is kept only in that Streamlit session.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and choose **Create app**.
3. Select this repository, branch, and streamlit_app.py.
4. Open **Advanced settings** and select Python 3.12.
5. Deploy. No server-side API secret is required.

The app opens without a key so visitors can review the landing page, but lesson
generation requires a visitor-supplied API key kept only in that session.

## Local smoke test

Run: python -m pip install -r requirements.txt
Then: python -m pytest -q
Then: python -m streamlit run streamlit_app.py

Research-only dependencies are intentionally isolated in
requirements-research.txt; they are not needed by the website.

## Security and cost

- api_keys.txt and .streamlit/secrets.toml are ignored by Git.
- A visitor-entered key is passed directly to the API client and is not written
  to disk or copied into process-wide environment variables.
- Illustrated MP4 generation is opt-in because it makes seven image calls.
- Gate 1 blocks image spending when the explanation remains unsafe after repair.
- Gate 2 reports whether the completed visuals are publish-ready.
