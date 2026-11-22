# Style library (样片库 / 风格记忆)

Each style in `agent_tools/styles.py` is defined by one sample photo in this folder. The agent
matches a photo's colour, exposure and tone curve to the sample (`match_reference`), so the sample
*is* the style.

Two samples ship with the repository (see [SOURCES.md](SOURCES.md)): `film.jpg` and
`epic_landscape.jpg`. For the other styles, add a photo you are allowed to use under the file name
below. Styles without a sample are skipped automatically.

| File | Style | What a good sample looks like |
|---|---|---|
| spring.jpg | 春 Spring | blossom against a clear sky, bright and fresh |
| summer.jpg | 夏 Summer | vivid colour, deep blue sky or sea, strong sun |
| autumn.jpg | 秋 Autumn | golden orange and red foliage |
| winter.jpg | 冬 Winter | clean snow whites, cool shadows |
| blue_hour.jpg | 蓝调 Blue hour | deep blue dusk or night, cool tones |
| night_portrait.jpg | 夜景人像 Night portrait | close portrait at night, dark background, bokeh |
| street_night.jpg | 街头夜景 Night street | person in a neon-lit street at night |
| street.jpg | 街景 Street | daylight city street portrait, muted tones |
| wedding_light.jpg | 婚礼浅色 Light wedding | bright, airy, pale, low saturation |

Tips: crop off watermarks, captions and black borders first, because they distort the colour
statistics. About 1000 px on the long side is enough.
