#!/bin/bash

PROJECT=$1
VERSION=$2
TOOL=$3  # evosuite
ID=$4    # test suite id
BUDGET=$5
SEED=$6
BUGGY_TMP_DIR=/tmp/${ID}/${PROJECT}-${VERSION}b
FIXED_TMP_DIR=/tmp/${ID}/${PROJECT}-${VERSION}f
TEST_SUITE_FILE=$EVOSUITE_TEST/$ID/${PROJECT}-${VERSION}.tar.bz2    # /root/results/evosuite_test/
TEST_SUITE_CONTENT=$EVOSUITE_TEST/$ID/${PROJECT}-${VERSION}/
EVOSUITE_CONFIG=$EVOSUITE_CONFIG/evosuite-config.$ID.$BUDGET.$SEED  # results/evosuite_config/

METADATA_DIR=$D4J_METADATA/${PROJECT}-${VERSION}     # root/results/metadata/Lang-1
FAILING_TESTS=$METADATA_DIR/tests.trigger
RELEVANT_CLASSES=$METADATA_DIR/classes.relevant
RELEVANT_METHODS_DIR=$METADATA_DIR/methods.relevant
COV_DIR=$METADATA_DIR/coverage

export EVOSUITE="java -jar /root/workspace/evosuite/evosuite-master-1.0.2.jar"

echo "\n************************************"
echo $PROJECT $VERSION
echo "************************************\n"

echo $BUGGY_TMP_DIR
[ -d "$BUGGY_TMP_DIR" ] && rm -rf $BUGGY_TMP_DIR

defects4j checkout -p ${PROJECT} -v ${VERSION}b -w $BUGGY_TMP_DIR   # checkout project to buggy_tmp_dir
if [ -d "$BUGGY_TMP_DIR" ]; then
  echo "Checkout succeed!"
else
  echo "Checkout failed"
  exit 1
fi

if [ -d "$METADATA_DIR" ]; then
  echo "$METADATA_DIR exists"
else
  mkdir $METADATA_DIR
fi

cd $BUGGY_TMP_DIR
defects4j export -p tests.trigger -o $FAILING_TESTS # List of test methods that trigger (expose) the bug
echo "defects4j export -p tests.trigger -o $FAILING_TESTS"
defects4j export -p classes.relevant -o $RELEVANT_CLASSES # Classes loaded by the JVM when executing all triggering tests

echo "Relevant classes"
cat $RELEVANT_CLASSES

echo "" >> $FAILING_TESTS
echo "" >> $RELEVANT_CLASSES

echo ""

# ---------------------------------------------------------------------------
if [ -d "$RELEVANT_METHODS_DIR" ]; then
  echo "$RELEVANT_METHODS_DIR exists"
else
  mkdir $RELEVANT_METHODS_DIR
fi

if [ -d "$COV_DIR" ]; then
  echo "$COV_DIR exists"
else
  mkdir $COV_DIR
fi

# measure coverage using TFD_EVOSUITE?
while IFS= read -r tc      # measure code coverage of failing tests
do
  cd $BUGGY_TMP_DIR
  COV_FILE="$COV_DIR/$tc.xml"
  echo $COV_FILE
  if [ -f "$COV_FILE" ]; then
    echo "$COV_FILE exists"
  else
    echo "Measuring the coverage of $tc..."
    defects4j coverage -t "$tc" -i $RELEVANT_CLASSES  # Measure code coverage for all classes listed in instrument_classes
    mv coverage.xml "$COV_DIR/$tc.xml"
  fi
done < $FAILING_TESTS

cd $D4J_EXPR  # $D4J_EXPR=/root/workspace 
python3.6 save_relevant_methods.py $COV_DIR $RELEVANT_METHODS_DIR  # calculate budgets and output relevant method by analyzing coverage file

# ---------------------------------------------------------------------------
cd $BUGGY_TMP_DIR

# TODO: change the command of EVOSUITE for generating tests
if [ -d "$TEST_SUITE_CONTENT" ]; then               # generate tests with EVOSUITE
  echo "test suite $TEST_SUITE_CONTENT exists"
else
  cd $BUGGY_TMP_DIR
  project_cp=$(defects4j export -p cp.compile -w $BUGGY_TMP_DIR) # Classpath to compile and run the project, i.e., /tmp/Lang-3b/target/classes

  if [ -f "$EVOSUITE_CONFIG" ]; then
    echo "config file exists"
  else
    echo "copying config file..."
    cp $EVOSUITE_DEFAULT_CONFIG $EVOSUITE_CONFIG  # /root/evosuite-config, /root/results/evosuite_config/
  fi
  echo "Evosuite Config: $EVOSUITE_CONFIG"

  [ ! -d $EVOSUITE_REPORT/${ID}/ ] && mkdir $EVOSUITE_REPORT/${ID}/
  [ ! -d $EVOSUITE_REPORT/${ID}/${PROJECT}/ ] && mkdir $EVOSUITE_REPORT/${ID}/${PROJECT}/
  for class in $(cat $RELEVANT_CLASSES); do      # generate test for relevant classes
    echo $class
    if [ -f "$RELEVANT_METHODS_DIR/$class" ]; then
      budget_ratio_file=$RELEVANT_METHODS_DIR/$class.budget
      budget_ratio=$(cat $budget_ratio_file)
      # echo $budget_ratio_file
      # echo $budget_ratio
      class_budget=$(python3.6 -c "from math import ceil; print(int(ceil($budget_ratio*$BUDGET)))")
      echo "-----------------------------------------------"
      echo "Generating test suite for class [$class] (w/ budget: ${class_budget}s)"
      echo "- Target methods:"
      cat $RELEVANT_METHODS_DIR/$class
      echo ""
      echo "-----------------------------------------------"
      echo "$EVOSUITE -class $class -projectCP $project_cp -Dsearch_budget=$class_budget -seed=$SEED -Dreport_dir=$EVOSUITE_REPORT -Dtest_dir=$TEST_SUITE_CONTENT -Dtarget_method_list=$(cat $RELEVANT_METHODS_DIR/$class | tr '\n' ':' | sed 's/:$//') $(cat ${EVOSUITE_CONFIG})"
      # $EVOSUITE -class $class -projectCP $project_cp -Dcriterion=line -generateTests -Dsearch_budget=$class_budget -seed=$SEED -Dreport_dir=$EVOSUITE_REPORT/${ID}/${PROJECT}/ -Dtest_dir=$TEST_SUITE_CONTENT -Dtarget_method_list=$(cat $RELEVANT_METHODS_DIR/$class | tr '\n' ':' | sed 's/:$//') $(cat ${EVOSUITE_CONFIG})
      $EVOSUITE -class $class -projectCP $project_cp -Dcriterion=line -generateTests -Dsearch_budget=$class_budget -seed=$SEED -Dreport_dir=$EVOSUITE_REPORT/${ID}/${PROJECT}/ -Dtest_dir=$TEST_SUITE_CONTENT $(cat ${EVOSUITE_CONFIG})
      cat $EVOSUITE_REPORT/${ID}/${PROJECT}/statistics.csv >> $EVOSUITE_REPORT/${ID}/${PROJECT}/statistics.${PROJECT}-${VERSION}.csv
      rm $EVOSUITE_REPORT/${ID}/${PROJECT}/statistics.csv
    else
      echo "no methods info"
    fi
  done;
fi

# ----------------------------------------------------------------------------------
if [ -d "$FIXED_TMP_DIR" ]; then
  echo "$FIXED_TMP_DIR exists"
else
  defects4j checkout -p ${PROJECT} -v ${VERSION}f -w $FIXED_TMP_DIR
  if [ -d "$FIXED_TMP_DIR" ]; then
    echo "Checkout succeed!"
  else
    exit 1
  fi
fi

# TODO: change the directory of tests
# get ground truth oracle and measure coverage of the generated tests
cd $D4J_EXPR && python3.6 measure_evosuite_coverage.py $PROJECT $VERSION -t $TOOL -i $ID

rm -rf $BUGGY_TMP_DIR
rm -rf $FIXED_TMP_DIR


# output: 
# Result files:
# - `./results/evosuite_test/<test_suite_id>/<pid>-<vid>`
#   - The generated test suite
# - `./results/evosuite_coverage/<test_suite_id>/<pid>-<vid>/`
#   - The coverage of the generated test cases
#   - Use `pandas.read_pickle` to load the coverage files.
# - `./results/evosuite_report/<test_suite_id>/statistics.<pid>-<vid>.csv`
#   - The EvoSuite report
# - `./results/evosuite_oracle/<test_suite_id>/<pid>-<vid>.failing_tests`
#   - A set of test cases that show different behaviours in the buggy and fixed versions