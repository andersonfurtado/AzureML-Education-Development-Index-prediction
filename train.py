
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('-f')
    
    #parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    #parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations to converge")

    parser.add_argument("--kernel", type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument("--penalty", type=int, default=100, help="Maximum number of iterations to converge")

    primary_metric_name='r2_score'
    args = parser.parse_args()

    #run.log("Regularization Strength:", np.float(args.C))
    #run.log("Max iterations:", np.int(args.max_iter))

    run.log("Regularization Strength:", np.float(args.kernel))
    run.log("Max iterations:", np.int(args.penalty))

    #model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    #joblib.dump(model,'outputs/model.joblib')
    #accuracy = model.score(x_test, y_test)

    model = LogisticRegression(kernel=args.kernel, penalty=args.penalty).fit(x_train, y_train)
    joblib.dump(model,'outputs/model.joblib')
    accuracy = model.score(x_test, y_test)
    
    run.log('r2_score', np.float(accuracy))
if __name__ == '__main__':
    main()
© 2021 GitHub, Inc.
