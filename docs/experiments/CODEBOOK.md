# Codebook: labelling Florence-2 object strings

You will see descriptions of single video frames from short social-media videos.
A description is the list of objects that an object detector (Florence-2)
found in the frame. You do **not** see the image. Judge only what the listed
objects most plausibly depict.

## The input

The detector's output was stored with the object names **glued together without
separators**. For example, `human facemantrousers` means the objects
*human face*, *man* and *trousers*. Each item shows:

* `raw`: the stored string, exactly as archived;
* `objects`: a suggested split into object names. It is automatic and can be
  wrong; trust `raw` if they disagree.

The object names come from a fixed detector vocabulary similar to Open Images.
Some names have a specific meaning in that vocabulary:

* **`tank`** is an *armoured military tank*, not a container. (`tank top` is a
  garment.)
* **`weapon`** is a generic weapon class (often a firearm). `rifle`, `handgun`,
  `shotgun`, `missile`, `rocket`, `cannon`, `bomb`, `sword` and `dagger` are
  more specific weapons.
* **`airplane`, `helicopter`, `jet ski`, `boat`, `ship`, `submarine`, `truck`,
  `car`, `land vehicle`, `vehicle`** say nothing on their own about military
  or civilian use.
* `No object detected.` means the detector found nothing.

## What to record for each item

1. **`categories`**: every equipment type present among the objects (zero or more):
   * `weapons`: any weapon (firearms, artillery, missiles, rockets used as weapons, bladed weapons, generic `weapon`);
   * `aircraft`: any aircraft (airplane, helicopter, drone, …);
   * `maritime`: any watercraft (boat, ship, submarine, jet ski, …). *Marine animals, paddles and swimming pools are not watercraft*;
   * `ground_vehicles`: any land vehicle (tank, truck, car, motorcycle, bus, train, generic vehicle, …).

   A **toy** version of an object (the object list includes `toy` and the
   object is plausibly the toy) does not count. Equipment shown on a `poster`,
   `television` or `computer monitor` **does** count, because the frame shows it.

2. **`primary`**: the single most salient equipment type among `categories`,
   or `none` if `categories` is empty. If several are present, choose the one
   that is most likely the subject of the frame. When unsure, use the study's
   label order: `weapons`, then `aircraft`, then `maritime`, then
   `ground_vehicles`. For example, a helicopter plus generic wheels is
   `aircraft`; a rifle carried by a person in front of a car is `weapons`.

3. **`military`**: do the objects indicate **military** equipment or a
   military setting?
   * `yes`: at least one object is inherently military (`tank`, `missile`,
     `cannon`, an armoured vehicle, a warship or submarine), or the
     combination strongly implies a military setting.
   * `no`: the objects indicate civilian use (e.g. `truck` + `vehicle
     registration plate`, `boat` + `swimming pool`, `airplane` + `luggage and
     bags`, a `car` among street objects), or there is no equipment at all.
   * `ambiguous`: equipment is present but could be military or civilian
     (e.g. `helicopter` alone, `weapon` alone, `rifle` + `person`).

   Judge `military` from the objects listed, not from what is common on social media.

4. **`confidence`**: 1 = guess, 2 = fairly sure, 3 = certain.

## Examples

| raw | objects | categories | primary | military | confidence |
|---|---|---|---|---|---|
| `tank` | tank | ground_vehicles | ground_vehicles | yes | 3 |
| `persontank top` | person, tank top | — | none | no | 3 |
| `car tiretruckwheel` | car tire, truck, wheel | ground_vehicles | ground_vehicles | ambiguous | 2 |
| `helicopter` | helicopter | aircraft | aircraft | ambiguous | 3 |
| `helicoptertoy` | helicopter, toy | — | none | no | 2 |
| `personweapon` | person, weapon | weapons | weapons | ambiguous | 3 |
| `missilerocket` | missile, rocket | weapons | weapons | yes | 3 |
| `boatperson` | boat, person | maritime | maritime | ambiguous | 2 |
| `marine mammal` | marine mammal | — | none | no | 3 |
| `human facemantrousers` | human face, man, trousers | — | none | no | 3 |
| `airplaneposter` | airplane, poster | aircraft | aircraft | ambiguous | 2 |

Do not use outside knowledge about the video, and do not try to guess what any
other system decided. Work through every item independently.
