import os

import yaml


def get_template():
    template_yaml = "configs/seq_optim/seq_optim_template_len_128.yaml"
    with open(template_yaml) as file:
        dict_file = yaml.full_load(file)
    return dict_file


def generate_config_files(seq_names):
    exp_names = []
    os.makedirs("configs/seq_optim/len_128/", exist_ok=True)
    for seq in seq_names:
        seq_name = f"seq_{seq:03d}"

        template_dict = get_template()
        template_dict["EXP_NAME"] += seq_name
        template_dict["MODELS"]["seqOpt"]["PARAMS"]["seq_path"] += f"{seq_name}.pth"
        template_dict["MODELS"]["seqOpt"]["PARAMS"][
            "optical_flow_presaved"
        ] += f"{seq_name}.pth"

        new_yaml = f"configs/seq_optim/len_128/{seq:03d}.yaml"
        with open(new_yaml, "w") as file:
            yaml.dump(template_dict, file)

        exp_names.append(f"{seq:03d}")
        print(seq_name)

    return exp_names


def write_fblearner_script(exp_names):
    f_path = "/tmp/seq_optim_flow2d__len_128.sh"
    with open(f_path, "w") as f:
        for exp_name in exp_names:
            print(exp_name)
            proc = [
                "flow-cli",
                "canary",
                "hmr.workflow.run_proc_wf@//fblearner/flow/projects/hmr:workflow",
            ]
            proc += ["--mode", "opt"]
            proc += ["--run-as-secure-group", "xr_people_zurich"]

            proc += ["--parameters-json"]
            cfg = '"cfg"'
            path = f'"manifold://xr_body/tree/personal/andreydavydov/configs/seq_optim/len_128/{exp_name}.yaml"'
            proc += [str("'{" + f"{cfg}:{path}" + "}'")]
            proc += ["--name", f"len_128__seq_{exp_name}"]
            proc = " ".join(proc)

            f.write(proc)
            f.write("\n")


if __name__ == "__main__":
    seq_names = list(range(253))

    ### 1. generate local config files
    exp_names = generate_config_files(seq_names)

    ### 2. write a bash script to send all experiments to flow
    write_fblearner_script(exp_names)

    # the rest is run from console
    ### 3. copy configs to remote
    ### 4. remove configs from local
    ### 5. run bash script from fbcode root
