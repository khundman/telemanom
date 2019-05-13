
docker build -t telemanom .

docker-compose up

docker-compose down
# -v --rmi all --remove-orphans
#docker rm -v $(docker ps -aq -f 'status=exited')
#docker rmi $(docker images -aq -f 'dangling=true')
