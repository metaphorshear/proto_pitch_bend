import pitch_bend as pb

class TestPitchBend:
  def test_vibrato(self):
    input = np.array([1, 1, 1, 1, 1, 1, 1])
    t = pb.vibrato(input, 100, 0.25)
    assert t != input
