# Bridge Data Robot infrastructure

Platform for data acquisition and training with WidowX in docker container.

### Project structure

'docker' folder - contains Dockerfile and docker-compose.yml file.  
'widowx_envs' folder - docker container-mounted python package.

### Environment variables

The following environment variables are used and can be adjusted to mount given directory in the docker container:

- CODE - mounted as '~/code' folder (defaults to 'code' folder in the repository root)
- DATA - mounted as '~/trainingdata' folder (defaults to 'trainingdata' folder in the repository root)
- EXP - mounted as '~/experiments' folder (defaults to 'experiments' folder in the repository root)

#### Important
In order to make sure the default values are used and required directories are created with user ownership keep using `scripts/docker-compose` proxy instead of plain `docker-compose` when running the container.

### Host machine setup

In order to set up the host machine to run robonetv2 run:

```shell script
bash scripts/host_install.sh
```

### Pull image and run container

To avoid specifying the arm type in docker container after each container recreation, set up ROBONETV2_ARM environment variable.  
For example:
```shell script
echo "export ROBONETV2_ARM=wx250s" >> ~/.bashrc && source ~/.bashrc
```

Then run the container.
```shell script
scripts/docker-compose \
   -f ./docker/docker-compose.yml \
   up \
   -d \
   --force-recreate
```

#### Build image and run container

In order to use the provided Dockerfile.base and rebuild the image with your changes use 'docker/docker-compose.build.yml' file:

```shell script
scripts/docker-compose \
   -f ./docker/docker-compose.build.yml \
   up \
   -d \
   --force-recreate \
   --build
```

### Access the container

```shell script
docker exec -it robonetv2_${USER} /bin/bash
```

`USER` variable is used to avoid the clashes between the containers of different users.

### Restart the container

```
scripts/docker-compose \
-f ./docker/docker-compose.yml \
up \
-d
```

### Cleanup

```
scripts/docker-compose \
   -f ./docker/docker-compose.yml \
   down \
   --rmi all \
   --volumes
``` 

### Run the code

#### Run the stack with one command
```
~/scripts/run.sh
```

For the available arguments of run.sh call `~/scripts/run.sh -h`

To verify the robot functionality call the exemplary training environment:

```
python ~/widowx_envs/widowx_envs/widowx/widowx_env.py
```

After finishing the interaction before the shutdown you can call `go_sleep` from bash to move the robot to the sleep position.

#### Data collection with teleoperation:
```
cd ~/widowx_envs
python widowx_envs/run_data_collection.py experiments/toykitchen_fixed_cam/conf.py
```
You can change data collection parameteres in `experiments/toykitchen_fixed_cam/conf.py`

#### Running an imitation learning policy
```
cd ~/widowx_envs
python widowx_envs/run_data_collection.py experiments/toy_kitchen_v0_control/conf_highres.py
```
You can change the policy rollout parameteres in `experiments/toykitchen_fixed_cam/conf.py`

#### Global parameters for WidowX envs

Each WidowX environment is reading 'widowx_envs/global_config.yml' created at the first run of 'scripts/run.sh'. That's a file which can be used as set of parameters
which will be used by all WidowX envs (including teleoperation). For example, this file can be used to rotate the workspace in all WidowX environments
on the specific machine by adding this line:
```
workspace_rotation_angle_z: 1.57
```

This configuration change will be also used by the `go_sleep` function. In order to avoid the changes in the configuration file from showing in git, please call:
```
git update-index --assume-unchanged widowx_envs/global_config.yml
```
To revert it, call:
```
git update-index --no-assume-unchanged widowx_envs/global_config.yml
```
`git update-index` does not propagate with git, and each user will have to run it independently.


#### Using realsense cameras

The realsense cameras require different drivers than RGB cameras.  If you are using realsenses, change the `camera_string` in `scripts/run.sh` to `realsense:=true`.

You will also need to update the device IDs in `/widowx_envs/widowx_envs/widowx/launch/realsense.launch` to match your cameras.

## Troubleshooting

##### Permission errors

If you run into following errors:

```
Traceback (most recent call last):
  File "urllib3/connectionpool.py", line 677, in urlopen
  File "urllib3/connectionpool.py", line 392, in _make_request
  File "http/client.py", line 1277, in request
  File "http/client.py", line 1323, in _send_request
  File "http/client.py", line 1272, in endheaders
  File "http/client.py", line 1032, in _send_output
  File "http/client.py", line 972, in send
  File "docker/transport/unixconn.py", line 43, in connect
PermissionError: [Errno 13] Permission denied
```
that can be fixed by running the following commands and subsequently restarting the PC (the log out and log back in is sometimes not sufficient):

```
sudo groupadd docker
sudo usermod -aG docker $USER
```
