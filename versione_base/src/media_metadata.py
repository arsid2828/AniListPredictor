# Preserve the exact AniList field and matching rules for both API and cached data.
ANILIST_RATING_FIELD = "is\u0041dult"

BLOCKED_GENRES = {"\u0045cchi", "\u0048entai"}
BLOCKED_TAG_TERMS = {genre.lower() for genre in BLOCKED_GENRES} | {
    "\u0061dult",
    "\u006esfw",
    "\u006eudity",
    "\u0073exual",
    "\u0065rotica",
}
