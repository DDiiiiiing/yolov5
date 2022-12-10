1. install nvidia-docker, follow https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html 

2. sudo docker pull ddiiing/yolov5:version1 
제 도커허브의 파일입니다. 다운받는데 엄청오래걸립니다.

3. sudo docker run -it --net host -v /dev/:/dev/ --name yolov5 ddiiing/yolov5:version1
다운받은 도커 이미지로부터 컨테이너를 생성합니다.
-it : interactive 모드. 커맨드 창을 활성화
--net host : 도커의 네트워크를 isolate하지 않습니다. 도커 밖의 roscore와 통신할 수 있게합니다
-v /dev/:/dev/ : 도커 밖의 /dev/ 와 내부의 /dev/ 를 공유합니다. 리얼센스를 인식할 수 있습니다. 
-v 
--name : yolov5 : 이 컨테이너의 이름은 yolov5 입니다. 바꿔도 무방합니다.
ddiiing/yolov5:version1 : 이 도커이미지로부터 컨테이너를 생성합니다.
컨테이너가 생성되면 root@{컴퓨터이름}:# 와 비슷한 곳으로 들어가 질겁니다

4. cd /home/catkin_ws
5. source devel/setup.bash 
일단 세팅은 끝났습니다.
6. (도커 밖에서!!!) roslaunch realsense2_camera rs_camera.launch 
다른 런치 파일 만든거 있음 그걸로 하셔도 됩니다. 저는 1280*720, 필터세팅 등 되어있는 런치파일 따로 만들어서  쓰고있습니다.
7. (도커 안에서!!!) roslaunch yolov5 predict_seg.launch weights:=fake conf_thres:=0.5
weights:=fake : 원래대로라면 학습 결과 파일(*.pt)의 절대주소를 넣어야 하는데 제가 fake, real 넣으면 그 둘의 파일 로드되게 바꿔놨습니다. 새로운 파일로 하시려면 그 파일을 도커 안에 넣고(sudo docker cp ... ...) 절대 경로를 넣어야 합니다. default는 fake, real 둘 다 아닌 다른 데이터셋 학습 결과이니 수동으로  넣어주세요. 귀찮으시면 launch 파일 수정하세용
conf_thres:=0.5 : 신뢰도 0.5 미만은 출력, 토픽발행 안합니다.
원본 파일은 https://github.com/DDiiiiiing/yolov5 인데 도커 안에 있는 파일(yolov5/segment/predict_seg.py)은 weights 부분 조금 수정되어있습니당. 런치파일 위치 yolov5/launch

8. 7번 작업에서 에러가 난다면 https://forums.developer.nvidia.com/t/issues-building-docker-image-from-ngc-container-nvcr-io-nvidia-pytorch-22-py3/209034/5 에 따라서 아래 환경변수 넣어보세요. 저는 이거 하니까 됐는데 문제 있음 연락주세용
(도커 안에서!!!) export PATH="${PATH}:/opt/hpcx/ompi/bin" \ export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

9. 나중에 다시 이 컨테이너를 열려면 
(도커 밖에서!!!) sudo docker start yolov5 
컨테이너를 실행상태로 만듭니다. 이름은 아까 컨테이너 이름 설정한 거입니당
(도커 밖에서!!!) sudo docker exec -it yolov5 /bin/bash
이 명령어는 도커 터미널 창을 하나 더 열고 싶을때도 사용합니다 
docker run 을 한번 더하면 ddiiing/yolov5:version1 이미지를 기반으로하는 컨테이너가 하나 더 만들어집니다!!! 용량 잡아먹어요