import os
import xml.etree.ElementTree as ET
from pathlib import Path

PATH = "Data/es25nov11_13"


def read_xml(path):
    ret = {}

    path_list = Path(path).rglob('*.xml')

    for xml_path in path_list:
        xml_path = str(xml_path)
        if 'clu' in xml_path or 'prm' in xml_path or 'fet' in xml_path:
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        """for child in root:
            print(child)
            for c in child:
                print(' ', c)"""

        acsys = root.find('acquisitionSystem')
        for child in acsys:
            print(child)

        spikes = root.find('spikeDetection')
        groups = spikes.find('channelGroups').findall('group')
        file_name = '.'.join(xml_path.split('\\')[-1].split('.')[:-1])

        for i, group in enumerate(groups):
            channels = group.find('channels')
            num_channels = len(channels.findall('channel'))
            if num_channels != 8:
                print(f"File {file_name} on shank {i + 1} has only {num_channels} recording sites")
            ret[f"{file_name}_{i + 1}"] = num_channels

    return ret
