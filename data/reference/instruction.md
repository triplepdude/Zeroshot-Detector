You are annotating items for a research study. Follow the codebook below exactly. Do not use any tools, do not read or search any files, and do not consult anything other than this message. For every item in the JSON list at the end, return one answer that echoes its item_id and raw exactly.

# Codebook v2: labelling Florence-2 object strings

*Version 2 (frozen before any annotation). Its SHA-256 is recorded in
`data/reference/raw/MANIFEST.json`.*

You will see descriptions of single frames from short social-media videos. A
description is the list of objects an object detector found in the frame. You do
**not** see the image. Judge only what the listed objects most plausibly show.

## 1. The input

The detector's output was stored with the object names **glued together without
separators**. For example, `human facemantrousers` means *human face*, *man*
and *trousers*. Each item shows:

* `raw`: the stored string;
* `objects`: a suggested split into object names. It is automatic and can be
  wrong; read `raw` when they disagree.

## 2. Label table (researcher decisions)

Use this table. For a label not in the table, decide by analogy with the
definitions below (e.g. a *sailboat* is watercraft, tier T1).

| Role | Labels |
|---|---|
| **weapons** (any weapon; never needs a military score) | `weapon` (generic), `rifle`, `handgun`, `shotgun`, `missile`, `cannon`, `sword`, `dagger`, `bow and arrow` |
| conditional weapons | `rocket` is a weapon only together with `missile`, `weapon` or `tank`; otherwise it is not equipment. `knife` is a weapon only together with another weapon or a `tank`/`missile`/`cannon`. `kitchen knife` is never a weapon. |
| **aircraft**, tier T2 (military-typical, dual-use) | `helicopter`, `airplane`, `aircraft` (generic) |
| **maritime**, tier T2 | `boat`, `watercraft` (generic) |
| **maritime**, tier T1 (everyday civilian) | `canoe`, `kayak`, `jet ski` |
| **ground_vehicles**, tier T3 (inherently military) | `tank` (an armoured military tank; *not* a container) |
| **ground_vehicles**, tier T2 | `truck`, `pickup truck` |
| **ground_vehicles**, tier T1 | `car`, `van`, `suv`, `taxi`, `limousine`, `bus`, `train`, `motorcycle`, `bicycle`, `ambulance`, `golf cart`, `snowplow`, `cart`, `wheelchair`, `segway`, `snowmobile`, `land vehicle` (generic) |
| generic `vehicle` | alone it creates **no** category |
| parent rule | `weapon`, `vehicle`, `land vehicle`, `aircraft`, `watercraft` are generic names. When a more specific label of the same family is also listed, the generic one names the same object and adds nothing (`helicopter` + `vehicle` = one aircraft). |
| parts (never create a category) | `wheel`, `car tire`, `tire`, `vehicle registration plate`, `seat belt`, `steering wheel`, `propeller`, `taillight`, `headlight`, `windshield wiper`, `rearview mirror`, `bicycle wheel` |
| not equipment | `parachute`, `balloon`, `kite`, `life jacket`, `paddle`, `surfboard`, `swimming pool`, `tank top` (a garment), marine animals (`marine mammal`, `harbor seal`, `dolphin`, `whale`, `shark`) |
| civilian cues | `luggage and bags`, `suitcase`, `swimming pool`, `fish`, fishing rods (may appear as `ing rod`), `surfboard`, `life jacket`, `swimwear` |
| military cues | a `tank`, `missile` or `cannon` anywhere in the list; a `helmet` together with a weapon |
| depiction | `poster`, `television`, `computer monitor`, `laptop`, `mobile phone`, `tablet computer`, `picture frame`, `billboard`: equipment listed with them may be shown on a screen or print. It still counts. |
| toys | `toy` listed with **exactly one** equipment item: that item is the toy and does **not** count. With two or more equipment items, count them. |
| persons | `person`, `man`, `woman`, `boy`, `girl`, `human face` |
| `No object detected.` | never shown to you |

## 3. What to record for each item

1. **`categories`**: every equipment category present (`weapons`, `aircraft`,
   `maritime`, `ground_vehicles`), possibly none, after applying the
   parent, part, toy and conditional rules.
2. **`primary`**: the single most salient category, or `none`. Apply these
   rules **in order**:
   1. *Armament.* If a weapon is listed with a vehicle (aircraft, maritime or
      ground) and **no person** label is listed, the weapon is the vehicle's
      armament: `primary` = the vehicle's category.
   2. *Inherently military.* If a T3 object (`tank`) is listed, its category
      is primary.
   3. Otherwise, if a weapon is listed, `primary` = `weapons`.
   4. Otherwise choose the vehicle category with the highest tier; break ties
      by the order `aircraft`, `maritime`, `ground_vehicles`.
3. **`scores`**: for **each vehicle category** in `categories`, a military
   score from 0 to 3:
   * **3**: a T3 object of that category; or the armament rule applies; or a T2
     object together with a military cue.
   * **2**: a T2 object with no cue either way; or a T1 object together with a
     military cue.
   * **1**: a T2 object together with a civilian cue.
   * **0**: only T1 or generic objects, and no military cue.
   Do not use base rates about social media; use only the listed objects.
4. **`depiction`**: `screen_or_print` if equipment is listed with a depiction
   label, `toy_or_model` if the toy rule removed the only equipment item, otherwise
   `real_or_unclear`.
5. **`confidence`**: 1 = guess, 2 = fairly sure, 3 = certain.

Echo each item's `item_id` and `raw` exactly.

## 4. Worked examples

None of these strings occurs in the data.

| raw | objects | categories | primary | scores | depiction |
|---|---|---|---|---|---|
| `tankposter` | tank, poster | ground_vehicles | ground_vehicles | ground_vehicles=3 | screen_or_print |
| `truckvehicle` | truck, vehicle | ground_vehicles | ground_vehicles | ground_vehicles=2 | real_or_unclear |
| `truckluggage and bags` | truck, luggage and bags | ground_vehicles | ground_vehicles | ground_vehicles=1 | real_or_unclear |
| `helicopterswimming pool` | helicopter, swimming pool | aircraft | aircraft | aircraft=1 | real_or_unclear |
| `vanweapon` | van, weapon | weapons, ground_vehicles | ground_vehicles (armament) | ground_vehicles=3 | real_or_unclear |
| `swordperson` | sword, person | weapons | weapons | — | real_or_unclear |
| `helmetpersonrifletruck` | helmet, person, rifle, truck | weapons, ground_vehicles | weapons | ground_vehicles=3 | real_or_unclear |
| `kayakperson` | kayak, person | maritime | maritime | maritime=0 | real_or_unclear |
| `jet skiperson` | jet ski, person | maritime | maritime | maritime=0 | real_or_unclear |
| `boattoy` | boat, toy | — | none | — | toy_or_model |
| `tankpersonflag` | tank, person, flag | ground_vehicles | ground_vehicles | ground_vehicles=3 | real_or_unclear |
| `persontank topwoman` | person, tank top, woman | — | none | — | real_or_unclear |

Work through every item independently. Do not try to guess what any other
system decided, and do not use tools or files: everything you need is in this
message.
