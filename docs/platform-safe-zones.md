# Platform Safe Zones for Vertical Video (9:16)

Reference for UI overlay areas on TikTok, YouTube Shorts, and Instagram Reels. All measurements are for a **1080x1920** frame.

## Per-Platform UI Overlays

### TikTok

| Zone | Pixels | % of frame | UI elements |
|------|--------|------------|-------------|
| Top | 0-160px | 0-8% | Username, sound label, "Following" indicator |
| Bottom | 1440-1920px | 75-100% | Caption, hashtags, music ticker, nav bar |
| Right | 840-1080px | 78-100% width | Like, comment, bookmark, share, music disc |
| Left | 0-120px (bottom only) | 0-11% width | Username, caption text (bottom-left) |

TikTok is the most aggressive platform. The right-side engagement buttons span roughly y=600 to y=1500. The bottom overlay height depends on caption length (short: ~250px, long with hashtags: ~480px).

### YouTube Shorts

| Zone | Pixels | % of frame | UI elements |
|------|--------|------------|-------------|
| Top | 0-200px | 0-10% | Status bar, "Shorts" label, search, camera |
| Bottom | 1470-1920px | 77-100% | Channel name, subscribe, title, music, progress bar |
| Right | 888-1080px | 82-100% width | Like, dislike, comment, share, remix, avatar |
| Left | mostly clear | - | ~48-60px margin recommended |

The description area expands to ~480px from bottom when tapped. Right-side buttons span roughly y=900 to y=1550.

### Instagram Reels

| Zone | Pixels | % of frame | UI elements |
|------|--------|------------|-------------|
| Top | 0-130px | 0-7% | Status bar, "Reels" header, camera icon |
| Bottom | 1550-1920px | 81-100% | Username, follow, caption, audio bar, nav |
| Right | 980-1080px | 91-100% width | Like, comment, share, save, menu |
| Left | mostly clear | - | Caption text appears bottom-left below y=1550 |

Instagram has the smallest bottom overlay of the three platforms. Right-side buttons span roughly y=600 to y=1550.

## Unified Cross-Platform Safe Zone

Taking the worst case from each platform:

```
Top margin:     200px   (y > 10.4%)    — YouTube Shorts is tallest
Bottom margin:  1440px  (y < 75.0%)    — TikTok starts lowest
Left margin:    50px    (x > 4.6%)     — all platforms similar
Right margin:   840px   (x < 77.8%)    — TikTok buttons extend widest

Safe rectangle: 790 x 1240 px
```

## Subtitle Placement

The lower-third zone where viewers naturally expect captions, above all platform UI:

```
Subtitle sweet spot:  y = 1100-1400px  (57%-73% of frame height)
```

This sits above TikTok's bottom overlay (worst case at y=1440) while staying in the natural reading zone. Keep horizontal extent within x=50 to x=840 to avoid right-side engagement buttons on TikTok.

## Sources

- TikTok ad specs and Strike Social safe zone guide
- Orson Lord safe zone overlays for Reels, TikTok, Shorts
- Kapwing YouTube Shorts safe zone reference
- Kreatli platform safe zone guides (2025-2026)
- Zeely TikTok safe zones guide
