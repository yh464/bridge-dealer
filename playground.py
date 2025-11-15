from conditional_dealer import *
# dealer = Simulator(dirname = 'example1')
c1 = Simulator(dirname = '1C')
gf_constraint,_ = parse_string('south 8+P')
c2_constraint,_ = parse_string('south 3-H 3-S 5+C/D')
d2_constraint,_ = parse_string('south 4H/S 5+C/D')
mm_constraint,_ = parse_string('south 5+C 4+D')
mm_constraint.shape_constraint.permute_suits = SuitPermuter([False, False, True, True])
os_constraint,_ = parse_string('south 3-H 3-S 3-D 5+C')

multi, _ = parse_string('north 6-9P 6+S 3-H 4-C 4-D')
multi.shape_constraint.permute_suits = SuitPermuter([True, True, False, False])