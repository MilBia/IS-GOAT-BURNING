"""Prompt sets for the zero-shot verifier (#151).

Fixed BEFORE any evaluation run and not tuned on the eval set; every set that is run is reported. The scene words are
generic (fire / flames / burning structure vs. lights, lamps, sunset, reflections, winter town square, snow); none of
them names the goat or anything specific to the eval frames.

Each set maps a class name to a list of prompts. A class embedding is the L2-normalised mean of its prompt embeddings
(CLIP-style template ensemble; a single prompt is used as is). ``FIRE_CLASSES`` lists the classes reduced into the
``fire`` logit, all other classes form ``no_fire``.
"""

FIRE_CLASSES = ("fire", "flames", "burning_structure")

MINIMAL = {
    "fire": ["a photo of fire."],
    "flames": ["a photo of flames."],
    "town_square": ["a photo of a town square."],
    "christmas_lights": ["a photo of christmas lights."],
    "street_lamps": ["a photo of street lamps at night."],
    "sunset": ["a photo of a sunset."],
    "snow": ["a photo of snow."],
}

_RICH_CLASSES = {
    "fire": "fire",
    "flames": "bright orange flames",
    "burning_structure": "a burning building or structure on fire",
    "christmas_lights": "christmas lights and decorations",
    "street_lamps": "street lamps glowing at night",
    "sunset": "a sunset or sunrise with an orange sky",
    "reflections": "reflections of lights on a wet or icy surface",
    "headlights": "car headlights and traffic lights at night",
    "town_square_day": "a normal winter town square during the day",
    "town_square_night": "a normal town square at night",
    "snow": "snow on the ground",
    "illuminated_object": "an illuminated sculpture or decoration",
}

RICH = {k: [f"a photo of {v}."] for k, v in _RICH_CLASSES.items()}

_TEMPLATES = (
    "a photo of {}.",
    "a webcam image of {}.",
    "a low resolution photo of {}.",
    "a blurry photo of {}.",
    "a surveillance camera image of {}.",
)

ENSEMBLE = {k: [t.format(v) for t in _TEMPLATES] for k, v in _RICH_CLASSES.items()}

PROMPT_SETS = {"minimal": MINIMAL, "rich": RICH, "ensemble": ENSEMBLE}
