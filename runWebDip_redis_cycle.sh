#!/bin/bash


while true; do
	date
	echo "Stopping redis"
	service redis-server stop
	echo "Deleting database"
	rm /var/lib/redis/dump.rdb
	echo "Waiting 2 minutes"
	sleep 120
	echo "Starting redis"
	service redis-server start
	echo "Waiting a day"
	sleep 86400
done

