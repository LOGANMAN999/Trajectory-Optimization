import numpy as np

CODE_TO_BODY = {
    1: "Mercury",
    2: "Venus",
    3: "Earth",
    4: "Mars",
    5: "Jupiter",
    6: "Saturn",
    7: "Uranus",
    8: "Neptune"
    
}

BODY_TO_CODE = {name: code for code, name in CODE_TO_BODY.items()}

RESONANCES = [
    (2,1),
    (3,1),
    (3,2),
]

MU = {
    "Sun":     1.32712440018e11,    # km^3/s^2
    "Mercury": 2.2032e4,
    "Venus":   3.2486e5,
    "Earth":   3.986004418e5,
    "Mars":    4.2828314258e4,
    "Jupiter": 1.26686534e8,
    "Saturn":  3.7931187e7,
    "Uranus":  5.794548e6,
    "Neptune": 6.836527e6
}

R_BODY = {
  "Mercury": 2439.7,
  "Venus":   6051.8,
  "Earth":   6378.1,
  "Mars":    3396.2,
  "Jupiter": 71492,
  "Saturn":  60268,
  "Uranus":  25559,
  "Neptune": 24764
}

A_ORBIT = {
  "Mercury": 0.387098 * 1.4959787e8,  # AU → km
  "Venus":   0.723332 * 1.4959787e8,
  "Earth":   1.000000 * 1.4959787e8,
  "Mars":    1.523679 * 1.4959787e8,
  "Jupiter": 5.20440  * 1.4959787e8,
  "Saturn":  9.5826   * 1.4959787e8,
  "Uranus": 19.2184   * 1.4959787e8,
  "Neptune":30.11     * 1.4959787e8
}

PLANET_PERIOD = {
    body: 2*np.pi * np.sqrt(A_ORBIT[body]**3 / MU['Sun'])
    for body in A_ORBIT
}