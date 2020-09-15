# 베이스 이미지로 ubuntu:16.04 사용 
FROM ubuntu:16.04

# 메인테이너 정보 (옵션)
MAINTAINER LEAF chanho960122@gmail.com

# 환경변수 설정 (옵션)
ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8

# 기본 패키지들 설치 및 Python 3.7 설치
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.7 python3.7-dev python3-pip python3-setuptools python3-wheel gcc
RUN apt-get install -y git

# pip 업그레이드
RUN python3.7 -m pip install pip --upgrade

# 여러분의 현재 디렉토리의 모든 파일들을 도커 컨테이너의 /Sentiment_Analysis 디렉토리로 복사
ADD . /Sentiment_Analysis

# 5000번 포트 개방
EXPOSE 5000

# 작업 디렉토리로 이동
WORKDIR /Sentiment_Analysis

# 작업 디렉토리에 있는 requirements.txt로 패키지 설치
RUN pip3 install -r requirements.txt

# 컨테이너에서 실행될 명령어. 컨테이거나 실행되면 prototype.py실행시킨다
CMD python prototype.py