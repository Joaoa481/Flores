"""Donde florece mi cielo: every story time (seconds). The single source for picture and sound.
72 bpm, 4/4: one beat = 0.8333 s, one bar = 3.3333 s."""
BEAT = 60 / 72
BAR = 4 * BEAT
def bar(b, beat=0.0): return (b - 1) * BAR + beat * BEAT     # bar 1 starts at 0

T_LIGHT  = 0.4             # the guiding light focuses from bokeh to a point
T_DESC   = bar(2)          # 3.33  it starts its descent
T_TOUCH  = bar(3, 2)       # 8.33  it touches the ground: the meadow ripples into focus
T_GROW   = bar(4)          # 10.0  the stem rises
T_LEAF   = bar(4, 3)       # 12.5  the leaves unfold
T_BUD    = bar(5, 1.25)    # 14.17 the bud is formed
T_OPEN   = bar(5, 2)       # 15.0  the petals swirl open
T_BLOOM  = bar(6)          # 16.67 bloom lands on the chord
T_CLOUD  = bar(7)          # 20.0  the cloud enters from the right
T_RAIN   = bar(7, 3)       # 22.5  rain
T_HUSH   = bar(8, 2.2)     # 25.5  near-silence
T_FLARE  = bar(9)          # 26.67 the light flares
T_GUST   = bar(9, .4)      # 27.0  gust from left to right
T_STARS  = bar(9, 2.2)     # 28.5  the cloud has landed as stars
T_LIFT   = bar(10)         # 30.0  the flower lifts again
T_FLASH  = bar(10, 2)      # 31.67 synchronous flash from the flower
T_FIELD  = bar(10, 2.4)    # 32.0  the meadow blooms (pull-back)
T_TITLE0 = bar(12)         # 36.67 the fireflies rise and swirl
T_TITLE  = bar(12, 2)      # 38.33 the title lands
T_OUT    = 44.0            # the title dims
DUR      = 45.0

TITLE = ['Donde tú estás,', 'la noche aprende a florecer.']
