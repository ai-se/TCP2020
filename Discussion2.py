import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import sys
# import statistics


def read_data(path):
    data = pd.read_csv(path)
    data = data.to_numpy()

    return data


def main():
    projects = ['TCT_deeplearning4j@deeplearning4j.csv', 'TCT_diaspora@diaspora.csv',
                'TCT_eclipse@jetty-project.csv',
                'TCT_facebook@presto.csv', 'TCT_galaxyproject@galaxy.csv', 'TCT_Graylog2@graylog2-server.csv',
                'TCT_jruby@jruby.csv', 'TCT_languagetool-org@languagetool.csv',
                'TCT_locomotivecms@engine.csv', 'TCT_loomio@loomio.csv', 'TCT_materialsproject@pymatgen.csv',
                'TCT_mdanalysis@mdanalysis.csv', 'TCT_middleman@middleman.csv', 'TCT_nutzam@nutz.csv',
                'TCT_ocpsoft@rewrite.csv',
                'TCT_openforcefield@openforcefield.csv', 'TCT_openSUSE@open-build-service.csv', 'TCT_parsl@parsl.csv',
                'TCT_puppetlabs@puppet.csv', 'TCT_radical-sybertools@radical.csv', 'TCT_rails@rails.csv',
                'TCT_reactionMechanismGenerator@RMG-Py.csv', 'TCT_rspec@rspec-core.csv', 'TCT_spotify@luigi.csv',
                'TCT_square@okhttp.csv', 'TCT_structr@structr.csv', 'TCT_thinkaurelius@titan.csv',
                'TCT_unidata@metpy.csv',
                'TCT_Unidata@thredds.csv', 'TCT_yt-project@yt.csv']

    project_name = ['deeplearning4j', 'diaspora', 'eclipse', 'facebook', 'galaxyproject',
                    'Graylog2',
                    'jruby', 'languagetool-org', 'locomotivecms', 'loomio', 'materialsproject', 'mdanalysis',
                    'middleman', 'nutzam',
                    'ocpsoft', 'openforcefield', 'openSUSE', 'parsl', 'puppetlabs', 'radical-sybertools', 'rails',
                    'reaction...', 'rspec', 'spotify', 'square', 'structr', 'thinkaurelius', 'unidata@metpy',
                    'Unidata@thredds', 'yt-project']

    final_result = []
    for index in range(len(projects)):
        file_path = 'data/' + projects[index]
        data = read_data(file_path)
        print("-------------------------------")
        print(projects[index])

        count_failure_time = 0
        count_fail_evi = 0

        for i in range(len(data)):
            for j in range(1, len(data[0])):
                if data[i][j] == "F":
                    count_failure_time += 1

                if i != 0:
                    if data[i][j] == "F":
                        if data[i - 1][j] == "F":
                            count_fail_evi += 1

        final_result.append([round(count_fail_evi / count_failure_time, 2), project_name[index]])

    final_result = sorted(final_result, reverse=True)
    final_value = []
    project_name = []

    for item in final_result:
        final_value.append(item[0])
        project_name.append(item[1])

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(8, 6))

    y_value = tuple(project_name)
    y_pos = np.arange(len(project_name))
    x_value = np.array(final_value)

    color = ['#D3D3D3' for i in range(30)]
    color.append('black')

    ax.barh(y_pos, x_value, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_value)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage of Consecutive Failed Test Cases')

    for i, v in enumerate(x_value):
        ax.text(v, i + .25, str(v), color='black', size=10)

    plt.show()


if __name__ == "__main__":
    main()
