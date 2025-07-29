from CompriskV2.model.ko3.utils import load_support_points_table, load_name_mapping


def test_load_support_points_table():
    path = "environment/dev/support_points/gasunie"
    df = load_support_points_table(path)
