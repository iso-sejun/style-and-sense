from pathlib import Path

import streamlit as st

from style_and_sense.config import DATA_DIR
from style_and_sense.storage import init_db, init_storage


def main() -> None:
    st.set_page_config(page_title="Style and Sense", layout="wide")

    init_storage()
    init_db()

    st.title("Style and Sense")
    st.caption("Choose a cute, complete outfit from clothes you already own.")

    st.success(f"Local closet storage is ready at `{DATA_DIR}`.")

    st.subheader("Build status")
    st.write(
        "Storage foundation is initialized. Next build bucket adds garment upload, "
        "editable metadata, and laundry status."
    )


if __name__ == "__main__":
    main()
