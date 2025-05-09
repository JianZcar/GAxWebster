import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from dataclasses import dataclass
from typing import List

@dataclass
class PhaseConfig:
    green: int
    amber: int
    all_red: int



def generate_tl_logic(
    phase_configs: [],
    tls_id: str = "J0",
    program_id: int = 1,
    offset: int = 0
) -> str:
    """
    Generate SUMO <tlLogic> XML for static traffic lights based on phase configurations,
    expanding the state string by repeating each character for each signal group count.

    :param phase_configs: List of PhaseConfig for each signal group.
    :param tls_id: Traffic light system ID.
    :param program_id: Program ID for the tlLogic.
    :param offset: Start offset in seconds.
    :return: Pretty-printed XML string.
    """
    additional = ET.Element('additional')
    tl_logic = ET.SubElement(
        additional,
        'tlLogic',
        id=tls_id,
        type="static",
        programID=str(program_id),
        offset=str(offset)
    )

    n_groups = len(phase_configs)

    for idx, pc in enumerate(phase_configs):
        # Build base state strings of length n_groups
        base_green = ['G' if i == idx else 'r' for i in range(n_groups)]
        base_amber = ['y' if i == idx else 'r' for i in range(n_groups)]
        base_all_red = ['r'] * n_groups

        # Expand each state char by repeating it n_groups times
        state_green = ''.join(char * n_groups for char in base_green)
        state_amber = ''.join(char * n_groups for char in base_amber)
        state_all_red = ''.join(char * n_groups for char in base_all_red)

        # Add phases
        ET.SubElement(tl_logic, 'phase', duration=str(pc.green), state=state_green)
        ET.SubElement(tl_logic, 'phase', duration=str(pc.amber), state=state_amber)
        ET.SubElement(tl_logic, 'phase', duration=str(pc.all_red), state=state_all_red)

    # Pretty print
    rough_string = ET.tostring(additional, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def main():
    # Define your phase configurations here
    phases = [
        PhaseConfig(green=82, amber=3, all_red=1),
        PhaseConfig(green=28, amber=3, all_red=1),
        PhaseConfig(green=41, amber=3, all_red=1)
    ]
    xml_output = generate_tl_logic(phases)

    # Save the XML output to a file
    output_filename = 'traffic_light.xml'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(xml_output)
    print(f"Traffic light logic saved to '{output_filename}'")

if __name__ == '__main__':
    main()

