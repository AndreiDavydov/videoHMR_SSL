import os

import yaml


def get_template():
    template_yaml = "configs/seq_optim/seq_optim__flow_temp_smooth_template.yaml"
    with open(template_yaml) as file:
        dict_file = yaml.full_load(file)
    return dict_file


def generate_config_files(flow_weights, smooth_weights):
    exp_names = []
    exp_paths = []
    os.makedirs("configs/seq_optim/grid_flow2d_smooth/", exist_ok=True)
    for flow_weight in flow_weights:
        for smooth_weight in smooth_weights:
            exp_name = f"grid_flow2d_{flow_weight}_smooth_{smooth_weight}"

            template_dict = get_template()
            template_dict["EXP_NAME"] += exp_name
            template_dict["LOSSES"]["flow_2d"]["WEIGHT"] = flow_weight
            template_dict["LOSSES"]["shape_smooth"]["WEIGHT"] = smooth_weight
            template_dict["LOSSES"]["j3d_smooth"]["WEIGHT"] = smooth_weight

            new_yaml = f"configs/seq_optim/grid_flow2d_smooth/{exp_name}.yaml"
            with open(new_yaml, "w") as file:
                yaml.dump(template_dict, file)

            exp_names.append(exp_name)
            exp_paths.append(new_yaml)
            print(flow_weight, smooth_weight)

    return exp_names


def write_fblearner_script(exp_names):
    f_path = "/tmp/seq_optim_grid_flow2d_smooth.sh"
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
            path = f'"manifold://xr_body/tree/personal/andreydavydov/configs/seq_optim/grid_flow2d_smooth/{exp_name}.yaml"'
            proc += [str("'{" + f"{cfg}:{path}" + "}'")]
            proc += ["--name", f"{exp_name}"]
            proc = " ".join(proc)

            f.write(proc)
            f.write("\n")


if __name__ == "__main__":
    flow_weights = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 300]
    smooth_weights = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 300]

    ### 1. generate local config files
    exp_names = generate_config_files(flow_weights, smooth_weights)

    ### 2. write a bash script to send all experiments to flow
    write_fblearner_script(exp_names)

    # the rest is run from console
    ### 3. copy configs to remote
    ### 4. remove configs from local
    ### 5. run bash script from fbcode root
