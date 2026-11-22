# Classical crop rules used by `propose_crops`

These are the standard framing rules taught in photography and film-framing
texts, for example Michael Freeman, *The Photographer's Eye*; Roy Thompson &
Christopher Bowen, *Grammar of the Shot*; Bryan Peterson, *Learning to See
Creatively*. They are written here in our own words, with the way each is
measured in code. They are guidelines: the tool scores them, the model decides.

## Rules for every photo

| Rule | What it says | How the tool measures it (feature) |
|---|---|---|
| Rule of thirds | Put the main point of interest on a third line or where two third lines cross (a "power point"), not dead centre, unless the scene is symmetric. | `placement`: distance from the focal point (eyes, or the centre of visual interest) to the nearest power point. |
| Clean edges | Nothing important should be sliced by the frame edge; bright or busy things at the edge pull the eye out. | `edge_clean`: visual interest in a thin band along the edges, compared with the frame average. |
| No amputated people | A partial person at the edge (half a face, an arm) is a distraction: include them properly or crop them out. | Big secondary faces must stay whole; small ones are penalised if partly cut. |
| Keep what matters | The crop should keep the subject and the things that tell the story. | `saliency_kept`: share of visual interest kept. |
| Symmetry exception | Symmetric scenes and reflections may be centred. | `symmetry` (left/right) and `reflection` (above/below the horizon) are reported. |

## People

| Rule | What it says | How it's measured |
|---|---|---|
| Eye line on the upper third | In portraits the eyes usually sit near the upper third line. | Part of `placement`. |
| Headroom | Leave a little space above the head; too little looks cramped. Extra space is fine in environmental portraits. | `headroom`: penalises only too little space above the head. |
| Don't crop at joints | Never let the frame edge fall on the neck, elbows/waist, wrists/hips, knees or ankles. Cut between joints: mid-chest, mid-thigh, mid-calf, or show the whole figure. | `joints`: where the bottom edge falls on the body, using classical figure proportions (a standing adult is about 8 head-heights tall: chin 1, elbows/waist 3, wrists/hips 4, knees about 6, ankles about 7.7). Approximate for sitting, leaning or bent poses. |
| Lead room (look room) | Leave more space in the direction the person is looking or moving than behind them. | `lead_room`: which way the face is turned (from the eye and nose landmarks) and the space ahead vs behind. |
| Shot size | Close-up, medium close-up (chest), medium (waist), cowboy (thigh), three-quarter, full body. Each is valid; choose deliberately. | `shot_size` label per candidate. |
| Orientation | A single standing person often suits a vertical frame (4:5, 3:4, 2:3), even when shot horizontally. | Vertical aspects are always offered when there is one main person. |

## Landscapes (no people, clear horizon)

| Rule | What it says | How it's measured |
|---|---|---|
| Horizon off-centre | Put the horizon on a third line, not through the middle (unless there is a reflection). | `horizon`: where the horizon falls in the crop. |
| Sky : land ratio | Give two-thirds of the frame to the more interesting part. A dramatic, textured sky gets about 2/3 (horizon on the lower third); a plain sky gets about 1/3 (horizon on the upper third), so the land gets the space. | Texture above vs below the horizon decides the target. |
| Wide formats | 3:2, 16:9 and 2:1 panoramas suit wide scenes; a 4:5 vertical suits a strong foreground leading into the scene. | Default aspect list for landscapes. |

## Known limits

- There's no body-pose model, so the joint rule is estimated from the face size and assumes an upright person.
- "Horizon" means the longest straight horizontal edge. Indoors it can be a table or a wall line, and the model should check it on the contact sheet.
- Leading lines aren't scored yet.

## Prominent objects (keep whole or leave out)
A strongly distinct non-person object (a lit lamp, a statue, a sign) is part of the story. A frame edge
must not slice it: keep it fully in, or leave it fully out (more than 2% and less than 90% of its box
inside = cut). Found with frequency-tuned saliency (Achanta 2009), with people removed by the person
segmenter. In check_composition, cutting an object of 5% of the frame or more is a hard violation.
For three-quarter and full-length shots the eye line may sit above the upper third (0.08-0.45 of the
height), because the whole figure has to fit below it.
