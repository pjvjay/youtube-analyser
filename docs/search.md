# Finding a person (`yta search`)

```bash
yta search "woman with purple hair talking" *.mp4 *.mkv *.webm
yta search "man in a blue jacket" *.mp4 --action walking
yta search "Jane smiling" *.mp4 --config my_people.toml
```

The query describes a person and, optionally, an action (any of the `--action`s above; without
one, every time the person is in view). An action alone ("people jumping", "anyone talking")
finds everyone doing it: the detector's own sections, as `yta detect --action` gives
them. The description can use:

- **hair colour**: black, brown, blond, red, grey, white, purple, pink, blue, green (and words
  like blonde, brunette, redhead, ginger, violet, silver, dark, light);
- **clothes colours**, of the top (shirt, jacket, coat, dress, suit, ...) or the bottom
  (trousers, jeans, skirt, ...), or just "in red" / "wearing blue" for either;
- **apparent gender** (woman, man, lady, guy, ...) and **age** (young, old, grey-haired);
- **voice**: "deep voice" or "high voice";
- **a name** you have enrolled (below).

Words it does not know are listed as ignored, so check the first line it prints (`looking for:
gender female, hair purple; action: talking`). The words, colours and thresholds are all in
`people.toml` (`yta config > my_people.toml` writes a copy): change what you need and pass it
with `--config` (only what differs is needed). `--explain` prints what was seen of every person and why they matched or not.

**How it works.** It reuses everything the detector (`yta detect`) saved in `--out` (the pose model's
output, faces, sound, camera movement; whatever is missing is computed and saved) and adds what
it measures itself (`appearance/<video>.npz`), so searching again, for anyone, is quick. The
steps go cheapest first, and a video is done as soon as nobody in it can match:

1. **The pose model**, as for every action.
2. **Hair and clothes colours**, if the query names any (`people.py`). Hair: the person's pixels
   (`src/youtube_analyser/assets/models/person_mask.onnx`) around the head, minus the face (its outline from the face landmark
   model) and minus anything the colour of their cheeks; only on faces with at least 10 px between
   the eyes. Top and bottom: the person's pixels between shoulders and hips, and hips and knees.
   Each colour is named by the nearest example in the palette (people.toml). If no sample of
   anyone has the colour asked for, nobody can match: done.
3. **People are followed through each shot**, linked from sample to sample as the walking rules
   link them (position after the camera's own movement, size, clothes), or by their box in
   close-ups; never across a cut, and not between two different hair colours (a cut the camera
   measurement missed). Every characteristic is then voted over all of a person's samples (at
   least 2, and the winner at least half of them), so one bad sample does not decide. People
   whose colours do not match are dropped.
4. **Their voice**, for gender, voice or a name: the talking rules (above) find which voice is
   each face's own. The voice's pitch gives "deep" (≤ 140 Hz) or "high" (≥ 155 Hz), and it
   decides apparent gender before the face does, because at 144p it is the more reliable of the
   two (below).
5. **Apparent gender and age from the face** (`src/youtube_analyser/assets/models/gender_age.onnx`, the slowest model), only
   for the people still in the running whose voice did not decide, and only on each one's 5
   clearest faces with at least 12 px between the eyes (`faces_per_person`).
6. **Each person is checked** against every part of the query: yes, no, or not seen well enough.
   Everyone with all yes is a **match**; with `--possible`, people with no "no" but something unseen
   are listed too, as **possible** (without it, the summary says how many there were).
7. **Their action** comes from the same rules as `--action` (talking, walking, ...), per person, and
   times they do it are joined into sections (as for the detector: gaps of up to 2 samples,
   intros left out unless `--keep-intro`).

**People by name.** Enrol someone with a stretch of video where only they talk (or a recording)
in your people.toml:

```toml
[[people]]
name = "Jane"
voice_from = { video = "20240105_sampleVid01.mp4", start = 120.0, end = 150.0 }
# or: voice_file = "jane.wav"
describe = "woman with purple hair"      # optional: used for when she is not talking
```

A person is Jane when their own voice matches hers: a fingerprint similarity of at least 0.86
(`same_voice_above`), and a pitch that does not rule it out. When they do not talk, `describe`
decides instead (all of it must be seen and match). Treat a voice match as "likely", not
certain: on meeting recordings the same speaker in two recordings scored 0.81-0.93 and different
speakers up to 0.90. In the test clips, Jane's voice in another video scored 0.92 and the other
voices 0.74-0.76.

**Output** (`--results`, default `results/search/<query>/`): `matches.csv` (video, person, start,
end, duration, action, match or possible, the evidence for each part of the query, YouTube link),
`index.html` with a thumbnail of each (the person boxed) and `thumbs/`. Local only, like the other
results.

**What to expect at 144p** (from tests on faces and voices shrunk or recorded at these sizes):

| Characteristic | Measured | Limits |
|---|---|---|
| Dyed hair (purple, green, pink, blue) | purple and green named right at 9 and 13 px between the eyes, pink and blue at 13 px | Needs a face turned to the camera; a hat or hood hides it |
| Natural hair | approximate | Ash blond and grey come out alike, as do dark brown and black; "dark" and "light" are safer words |
| Clothes | the blue and beige coats in the street clips named right (after adding two examples of beige in shade to the palette) | Colours shift with light and white balance: add examples to the palette from your videos (`--explain` prints what was measured) |
| Apparent gender by face | 88-100% right on 17 test faces with 13 px or more between the eyes, 82% at 9 px | Unreliable alone: a woman in sunglasses and one with dyed purple hair came out as men. In the test clips the woman scored 0.53-0.67 and the man 0.38, so only one was decided by face. Undecided between 0.25 and 0.6 |
| Apparent gender by voice | women 168-250 Hz, men 98-119 Hz in 12 test voices | Only for people seen talking; deep female and high male voices fall in between (not decided) |
| Age | not measured (the test faces had no known ages) | Treat as rough; only "young" (< 30) and "old" (> 55) are searchable |
| Names | see above | Only when they talk, or by `describe` |

Gender here is the apparent gender of a face or voice, as these models estimate it, not anyone's
identity; the words a query can use for it are in `people.toml`.

Speed: see [the channel search](channel.md#speed), which measured each
step. Gender, voice and name need the talking inputs (the face and sound passes, shared with
`--action talking`), which are the slowest part when someone might match.

