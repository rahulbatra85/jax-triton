clear

set -x


ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

bash ./scripts/clean.sh 2>&1 |tee $LOG_DIR/clean.log
bash ./scripts/install.sh 2>&1 |tee $LOG_DIR/install.log
bash ./scripts/build.sh 2>&1 |tee $LOG_DIR/build.log
bash ./scripts/test.sh 2>&1 |tee $LOG_DIR/test.log