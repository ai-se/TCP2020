import pandas as pd
# import numpy as np
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

    for index in range(len(projects)):
        file_path = 'data/' + projects[index]
        data = read_data(file_path)
        print("-------------------------------")
        print(projects[index])

        max_fail = 0
        min_fail = float('inf')
        fail_number = []

        for row in data:
            temp_row = row[1:len(row)]
            cur_failure = 0

            for tc in temp_row:
                if tc == "F":
                    cur_failure += 1

            if cur_failure > max_fail:
                max_fail = cur_failure

            if cur_failure < min_fail:
                min_fail = cur_failure

            fail_number.append(cur_failure)

        range_10 = int(len(fail_number) * 0.1)
        range_30 = int(len(fail_number) * 0.3)
        range_50 = int(len(fail_number) * 0.5)
        range_70 = int(len(fail_number) * 0.7)
        range_90 = int(len(fail_number) * 0.9)

        print("10%: " + str(sorted(fail_number)[range_10]))
        print("30%: " + str(sorted(fail_number)[range_30]))
        print("50%: " + str(sorted(fail_number)[range_50]))
        print("70%: " + str(sorted(fail_number)[range_70]))
        print("90%: " + str(sorted(fail_number)[range_90]))


if __name__ == "__main__":
    main()
