import matplotlib.pyplot as plt
from jaxspec.analysis.compare import plot_corner_comparison


def test_plot_ppc(get_individual_results, get_joint_result):
    for result in get_individual_results + get_joint_result:
        result.plot_ppc(percentile=(5, 95))
        plt.show()


def test_plot_corner(get_individual_results, get_joint_result):
    for result in get_individual_results + get_joint_result:
        result.plot_corner()
        plt.show()


def test_table(get_individual_results, get_joint_result):
    for result in get_individual_results + get_joint_result:
        print(result.table())


def test_compare(get_individual_results, get_joint_result):
    plot_corner_comparison(
        {name: res for name, res in zip(["PN", "MOS1", "MOS2", "ALL"], get_individual_results + get_joint_result)}
    )
    plt.show()
