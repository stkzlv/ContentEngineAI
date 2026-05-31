# Platform Safe Zones for Vertical Video (9:16)

Last updated: 2026-05-31

Canonical reference for UI overlay areas on TikTok, YouTube Shorts, and
Instagram Reels. All measurements are for a **1080x1920** frame. This is the
single source of truth for safe-zone numbers in the project. The subtitle and
promotional best-practices docs cite this file rather than carry their own
copies, and the runtime `PlatformSafeZone` defaults
(`src/video/config/constants.py`) track the unified rect below.

**Related docs**:
- [subtitle-best-practices.md](subtitle-best-practices.md) -- caption design;
  defers to this file for placement bounds.
- [promotional-video-best-practices.md](promotional-video-best-practices.md)
  -- promo strategy; defers to this file for the disclosure-corner and CTA
  placement zones.

## What changed in 2026

Two platform updates moved the numbers since the first version of this doc:

- **Meta unified Reels and Stories safe zones (March 2026)** to **14% top /
  35% bottom / 6% sides**. The 35% bottom is the big one: Reels now reserves
  ~670px at the bottom for the caption, follow button, and audio label, far
  more than the older ~320px pixel guidance.
- **TikTok added an "Add to Playlist" button (January 2026)** at bottom-right,
  widening the right-side dead zone by ~20px.

The net effect: the top margin is now driven by Instagram (270px), and the
bottom margin is driven by Instagram too (670px), not by TikTok as before.

## Per-platform UI overlays

Numbers are the reserved (unsafe) zones. Ranges reflect organic vs. ad/expanded
states; design to the larger value.

### TikTok

| Zone | Pixels | % of frame | UI elements |
|------|--------|------------|-------------|
| Top | 0-140px | 0-7% | Username, sound label, "Following" indicator |
| Bottom | 1440-1920px | 75-100% | Caption, hashtags, music ticker, nav (organic ~320px; with CTA/long caption up to ~480px) |
| Right | 900-1080px | 83-100% width | Like, comment, bookmark, share, music disc, playlist button (Jan 2026) |
| Left | 0-60px | 0-6% width | Username, caption text (bottom-left) |

Right-side buttons span roughly y=600 to y=1500.

### YouTube Shorts

| Zone | Pixels | % of frame | UI elements |
|------|--------|------------|-------------|
| Top | 0-120px | 0-6% | Status bar, "Shorts" label, search |
| Bottom | 1620-1920px | 84-100% | Channel name, subscribe, title, music, progress bar (expands to ~400px when description open) |
| Right | 930-1080px | 86-100% width | Like, dislike, comment, share, remix, avatar |
| Left | mostly clear | - | ~50-60px margin recommended |

Lightest UI of the three, but still center-biased. Right buttons span roughly
y=900 to y=1550.

### Instagram Reels (Meta unified, March 2026)

| Zone | Pixels | % of frame | UI elements |
|------|--------|------------|-------------|
| Top | 0-270px | 0-14% | Status bar, "Reels" header, camera icon |
| Bottom | 1250-1920px | 65-100% | Username, follow, caption, audio bar, nav (Meta unified 35% interactive zone) |
| Right | 960-1080px | 89-100% width | Like, comment, share, save, menu |
| Left | mostly clear | - | ~60px margin |

Reels now has the **largest** top and bottom reserves of the three platforms,
a reversal from earlier guidance where its bottom was the smallest.

## Unified cross-platform safe zone

Worst case from each side. A single render that has to work on all three
platforms must stay inside this rectangle:

```
Top margin:     270px   (y > 14.1%)   -- Instagram Reels (Meta unified 14%)
Bottom margin:  670px   (y < 65.1%)   -- Instagram Reels (Meta unified 35%)
Left margin:    60px    (x >  5.6%)   -- all platforms ~60px
Right margin:   180px   (x < 83.3%)   -- TikTok engagement column + playlist

Safe rectangle: 840 x 980 px, spanning x=60..900, y=270..1250
```

The runtime defaults in `src/video/config/constants.py` are the
fractions for these bounds (`SAFE_ZONE_MIN_X/MAX_X/MIN_Y/MAX_Y`). Keep the two
in sync; a follow-up issue tracks updating the constants to this 2026 union.

## Subtitle and text placement

The reading zone for captions, above all platform bottom UI and below the top
header:

```
Caption band:    y = 900-1150px   (47%-60% of frame height)
Block center:    y ~ 1000px       (~52%)
Hard floor:      y = 1250px       (65%) -- nothing important below this on Reels
Horizontal:      x = 60..900px    -- avoids TikTok right-side buttons
```

Center-ish placement (around 52%) beats a strict lower-third because the bottom
35% is now interactive UI on Reels and the bottom 25% on TikTok. Keep the
lowest caption pixel above y=1250. The disclosure overlay (`#ad`) goes
top-left or top-right inside the 270px top band, not bottom, so it clears the
caption zone and the centre-upper hook overlay.

## Sources

- [Kreatli -- Safe Zone Hub 2026 (TikTok / Reels / Shorts)](https://kreatli.com/guides/safe-zone-guide)
- [Kreatli -- TikTok Safe Zone 2026](https://kreatli.com/guides/tiktok-safe-zone)
- [Kreatli -- Instagram Reels Safe Zone 2026](https://kreatli.com/guides/instagram-reels-safe-zone)
- [Kreatli -- YouTube Shorts Safe Zone 2026](https://kreatli.com/guides/youtube-shorts-safe-zone)
- [behaviour.digital -- Meta Reels Safe Zone 14% top / 35% bottom / 6% sides (2026)](https://behaviour.digital/post/meta-reels-safe-zone-14-top-35-bottom-6-sides-the-2026-official-guide)
- [Zeely -- TikTok safe zones 2026](https://zeely.ai/blog/tiktok-safe-zones/)
- [Postplanify -- Social media safe zones 2026](https://postplanify.com/blog/social-media-safe-zones-2026-complete-guide)
