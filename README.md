# Style and Sense

Style and Sense is a hybrid multimodal RAG closet assistant for students. The MVP recommends complete daily outfits from clothing a user already owns, using local closet storage, image embeddings, style-rule retrieval, and structured LLM output.

## MVP Scope

- Streamlit app for garment upload, closet review, outfit requests, and saved outfit history.
- Local SQLite database for garment metadata, recommendations, and feedback.
- Local file storage for uploaded garment images.
- Local FAISS indexes for garment and style-rule retrieval in later build buckets.
- OpenAI is used for final outfit composition and explanation from retrieved garment IDs and style rules.

The MVP treats Streamlit storage as ephemeral demo storage. A production version would add user accounts and durable object storage.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
