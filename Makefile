NAME = lr
CONTAINER_PATH = "/tmp/visualisation"
HOST_PATH = "$(shell pwd)/visualisation"


build:
		docker build -t $(NAME) .

run:
		docker run -it $(NAME)

# run with copying visualisation files from container to host
run-v:
		docker run -v $(HOST_PATH):$(CONTAINER_PATH) -it lr

clean:
		docker system prune -a -f

re:
		make build
		make run-v

new:
		make clean
		make build
		make run-v