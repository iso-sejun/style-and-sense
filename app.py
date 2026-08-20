from pathlib import Path

import streamlit as st

from style_and_sense.config import DATA_DIR
from style_and_sense.metadata import (
    FashionClipTagger,
    MetadataSuggestion,
    MetadataSuggestionError,
    fallback_metadata_suggestion,
)
from style_and_sense.retrieval import build_garment_index, load_garment_index
from style_and_sense.storage import (
    count_garment_embeddings,
    create_garment,
    delete_garment,
    init_db,
    init_storage,
    list_garments,
    save_uploaded_image_bytes,
    update_garment,
)
from style_and_sense.style_rules import load_style_rules
from style_and_sense.vocab import (
    CATEGORIES,
    COLORS,
    FORMALITY_LEVELS,
    LAUNDRY_STATUSES,
    SEASON_TAGS,
    STYLE_TAGS,
    SUBCATEGORIES,
)


@st.cache_resource(show_spinner=False)
def get_fashion_clip_tagger() -> FashionClipTagger:
    return FashionClipTagger()


def option_index(options: list[str], value: str | None) -> int:
    if value in options:
        return options.index(value)
    return 0


def valid_defaults(options: list[str], values: list[str]) -> list[str]:
    return [value for value in values if value in options]


def get_metadata_suggestion(image_bytes: bytes, filename: str) -> MetadataSuggestion:
    cache_key = f"{filename}:{len(image_bytes)}"
    cached = st.session_state.get("metadata_suggestion")
    if cached and cached["key"] == cache_key:
        return cached["suggestion"]

    try:
        with st.spinner("Auto-tagging with FashionCLIP..."):
            suggestion = get_fashion_clip_tagger().suggest(image_bytes, filename)
    except MetadataSuggestionError as exc:
        st.warning(f"{exc} Using color and filename fallback for now.")
        suggestion = fallback_metadata_suggestion(image_bytes, filename)
    except Exception as exc:
        st.warning(
            "FashionCLIP suggestions are unavailable right now. "
            f"Using color and filename fallback instead. Details: {exc}"
        )
        suggestion = fallback_metadata_suggestion(image_bytes, filename)

    st.session_state["metadata_suggestion"] = {
        "key": cache_key,
        "suggestion": suggestion,
    }
    return suggestion


def empty_metadata_suggestion() -> MetadataSuggestion:
    return MetadataSuggestion(
        category="top",
        subcategory="t-shirt",
        colors=[],
        style_tags=[],
        season_tags=[],
        formality="casual",
        caption="",
    )


def rebuild_garment_index() -> int:
    tagger = get_fashion_clip_tagger()
    if not hasattr(tagger, "encode_image"):
        get_fashion_clip_tagger.clear()
        tagger = get_fashion_clip_tagger()
    index = build_garment_index(embed_image=tagger.encode_image)
    return len(index.garment_ids)


def render_upload_form() -> None:
    st.subheader("Add clothes")
    uploaded_file = st.file_uploader(
        "Upload a garment photo",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    image_bytes = uploaded_file.getvalue() if uploaded_file else None
    suggestion = (
        get_metadata_suggestion(image_bytes, uploaded_file.name)
        if image_bytes
        else empty_metadata_suggestion()
    )
    if uploaded_file:
        st.caption("Tags are auto-suggested. Review and fix anything that looks off.")

    with st.form("upload_garment_form", clear_on_submit=True):
        category = st.selectbox(
            "Category",
            CATEGORIES,
            index=option_index(CATEGORIES, suggestion.category),
        )
        subcategory = st.selectbox(
            "Subcategory",
            SUBCATEGORIES,
            index=option_index(SUBCATEGORIES, suggestion.subcategory),
        )
        colors = st.multiselect(
            "Colors",
            COLORS,
            default=valid_defaults(COLORS, suggestion.colors),
        )
        style_tags = st.multiselect(
            "Style tags",
            STYLE_TAGS,
            default=valid_defaults(STYLE_TAGS, suggestion.style_tags),
        )
        season_tags = st.multiselect(
            "Season tags",
            SEASON_TAGS,
            default=valid_defaults(SEASON_TAGS, suggestion.season_tags),
        )
        formality = st.selectbox(
            "Formality",
            FORMALITY_LEVELS,
            index=option_index(FORMALITY_LEVELS, suggestion.formality),
        )
        laundry_status = st.radio(
            "Laundry status",
            LAUNDRY_STATUSES,
            format_func=lambda value: value.replace("_", " ").title(),
            horizontal=True,
        )
        favorite = st.checkbox("Favorite item")
        caption = st.text_input(
            "Short description",
            value=suggestion.caption if uploaded_file else "",
            placeholder="white cropped tee, black straight-leg jeans, etc.",
        )
        submitted = st.form_submit_button("Add to closet")

    if not submitted:
        return

    if uploaded_file is None:
        st.error("Upload a garment photo before adding it to your closet.")
        return

    if not colors:
        st.error("Choose at least one color so recommendations have useful metadata.")
        return

    image_path = save_uploaded_image_bytes(
        image_bytes,
        original_filename=uploaded_file.name,
    )

    garment_id = create_garment(
        image_path=str(image_path),
        original_filename=uploaded_file.name,
        category=category,
        subcategory=subcategory,
        colors=colors,
        style_tags=style_tags,
        season_tags=season_tags,
        formality=formality,
        caption=caption.strip() or None,
        laundry_status=laundry_status,
        favorite=favorite,
    )
    try:
        rebuild_garment_index()
    except Exception as exc:
        st.warning(f"Added garment, but could not update the vector index yet: {exc}")
    st.success(f"Added garment `{garment_id}` to your closet.")
    st.rerun()


def render_garment_editor(garment: dict) -> None:
    form_key = f"edit_{garment['id']}"
    image_path = Path(garment["image_path"])

    if st.button("Auto-tag item", key=f"{form_key}_auto_tag"):
        if not image_path.exists():
            st.error("The image file for this garment is missing.")
        else:
            suggestion = get_metadata_suggestion(
                image_path.read_bytes(),
                garment["original_filename"] or image_path.name,
            )
            update_garment(
                garment["id"],
                category=suggestion.category,
                subcategory=suggestion.subcategory,
                colors=suggestion.colors,
                style_tags=suggestion.style_tags,
                season_tags=suggestion.season_tags,
                formality=suggestion.formality,
                caption=suggestion.caption,
                laundry_status=garment["laundry_status"],
                favorite=garment["favorite"],
            )
            st.success("Auto-tags applied. Review them below.")
            st.rerun()

    with st.form(form_key):
        category = st.selectbox(
            "Category",
            CATEGORIES,
            index=option_index(CATEGORIES, garment["category"]),
            key=f"{form_key}_category",
        )
        subcategory = st.selectbox(
            "Subcategory",
            SUBCATEGORIES,
            index=option_index(SUBCATEGORIES, garment["subcategory"]),
            key=f"{form_key}_subcategory",
        )
        colors = st.multiselect(
            "Colors",
            COLORS,
            default=[color for color in garment["colors"] if color in COLORS],
            key=f"{form_key}_colors",
        )
        style_tags = st.multiselect(
            "Style tags",
            STYLE_TAGS,
            default=[tag for tag in garment["style_tags"] if tag in STYLE_TAGS],
            key=f"{form_key}_style_tags",
        )
        season_tags = st.multiselect(
            "Season tags",
            SEASON_TAGS,
            default=[tag for tag in garment["season_tags"] if tag in SEASON_TAGS],
            key=f"{form_key}_season_tags",
        )
        formality = st.selectbox(
            "Formality",
            FORMALITY_LEVELS,
            index=option_index(FORMALITY_LEVELS, garment["formality"]),
            key=f"{form_key}_formality",
        )
        laundry_status = st.radio(
            "Laundry",
            LAUNDRY_STATUSES,
            index=option_index(LAUNDRY_STATUSES, garment["laundry_status"]),
            format_func=lambda value: value.replace("_", " ").title(),
            horizontal=True,
            key=f"{form_key}_laundry",
        )
        favorite = st.checkbox(
            "Favorite",
            value=garment["favorite"],
            key=f"{form_key}_favorite",
        )
        caption = st.text_input(
            "Description",
            value=garment["caption"] or "",
            key=f"{form_key}_caption",
        )

        save, remove = st.columns(2)
        save_clicked = save.form_submit_button("Save changes")
        remove_clicked = remove.form_submit_button("Delete")

    if save_clicked:
        if not colors:
            st.error("Each garment needs at least one color.")
            return
        update_garment(
            garment["id"],
            category=category,
            subcategory=subcategory,
            colors=colors,
            style_tags=style_tags,
            season_tags=season_tags,
            formality=formality,
            caption=caption.strip() or None,
            laundry_status=laundry_status,
            favorite=favorite,
        )
        st.success("Garment updated.")
        st.rerun()

    if remove_clicked:
        delete_garment(garment["id"])
        try:
            rebuild_garment_index()
        except Exception as exc:
            st.warning(f"Deleted garment, but could not refresh the vector index: {exc}")
        st.success("Garment deleted.")
        st.rerun()


def render_closet() -> None:
    garments = list_garments()

    st.subheader("Your closet")
    if not garments:
        st.info("Upload a few clothes to start building your closet.")
        return

    available_count = sum(
        garment["laundry_status"] == "available" for garment in garments
    )
    laundry_count = len(garments) - available_count
    st.caption(
        f"{len(garments)} items total | {available_count} available | "
        f"{laundry_count} in laundry"
    )

    for row_start in range(0, len(garments), 3):
        columns = st.columns(3)
        for column, garment in zip(columns, garments[row_start : row_start + 3]):
            with column:
                st.image(garment["image_path"], use_container_width=True)
                title = garment["caption"] or garment["subcategory"] or garment["category"]
                st.markdown(f"**{title}**")
                status = garment["laundry_status"].replace("_", " ").title()
                st.caption(
                    f"{garment['category']} | {', '.join(garment['colors'])} | {status}"
                )
                with st.expander("Edit item"):
                    render_garment_editor(garment)


def render_sidebar() -> None:
    rules = load_style_rules()
    categories = sorted({rule.category for rule in rules})
    garment_index = load_garment_index()
    st.sidebar.header("Project status")
    st.sidebar.metric("Style rules", len(rules))
    st.sidebar.metric(
        "Indexed garments",
        len(garment_index.garment_ids) if garment_index else count_garment_embeddings(),
    )
    st.sidebar.caption("Categories: " + ", ".join(categories))
    if st.sidebar.button("Rebuild garment index"):
        try:
            with st.spinner("Embedding closet with FashionCLIP..."):
                indexed_count = rebuild_garment_index()
            st.sidebar.success(f"Indexed {indexed_count} garments.")
        except Exception as exc:
            st.sidebar.error(f"Could not rebuild garment index: {exc}")


def main() -> None:
    st.set_page_config(page_title="Style and Sense", layout="wide")

    init_storage()
    init_db()
    render_sidebar()

    st.title("Style and Sense")
    st.caption("Choose a cute, complete outfit from clothes you already own.")

    st.success(f"Local closet storage is ready at `{DATA_DIR}`.")

    left, right = st.columns([1, 2])
    with left:
        render_upload_form()
    with right:
        render_closet()


if __name__ == "__main__":
    main()
