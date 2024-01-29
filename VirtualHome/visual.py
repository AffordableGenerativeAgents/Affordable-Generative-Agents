import sys

sys.path.append("<VirtualHome project Path>")
sys.path.append("<VirtualHome project Path>/virtualhome/simulation")

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication


script = ['<char0> [walk] <desk> (110)', '<char0> [walk] <chair> (109)',
          '<char0> [sit] <chair> (109)']  # Add here your script

print('Starting Unity...')
comm = UnityCommunication(
    file_name='<the path of the UnitySimulator executable>',
    x_display=1)

print('Starting scene...')
comm.reset()
comm.add_character('Chars/Female1')
print('Generating video...')
a, b = comm.render_script(script, recording=True, find_solution=True, output_folder='./')
print('Generated, find video in simulation/unity_simulator/output/')
