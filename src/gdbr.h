#if !defined(ARMA_DEFAULT_OSTREAM)
#define ARMA_DEFAULT_OSTREAMR Rcpp::Rcout
#endif

// Just to speed it up by removing unnecessary assertions. This should
// be commented out during debugging.
#define ARMA_NO_DEBUG 1

#include <RcppArmadillo.h>

struct gdbr_result;
gdbr_result gdbrcpp(const arma::mat& D, const arma::rowvec& y);

/**
 * There are several calculations that don't seem to be directly
 * supported by the armadillo matrix/vector classes. These functions
 * are simply here to support the calculations used in the GDBR calculation.
 *
 */

/** Simply multiples a rowvector by a colvector to produce a matrix
 *
 *  multiply two vectors and returns their product as a matrix
 *  comparable to the following calculation in R
			> a=c(1,2,3,4)
			> a %*% t(a)
			     [,1] [,2] [,3] [,4]
			[1,]    1    2    3    4
			[2,]    2    4    6    8
			[3,]    3    6    9   12
			[4,]    4    8   12   16
  */
arma::mat vecMul(const arma::rowvec& a, const arma::colvec& b);


/** Subtract a column vector from each column of a matrix
 *
 *  Similar to the follow R code:
			> a=c(1,2,3,4)
			> a %*% t(a) - a
			     [,1] [,2] [,3] [,4]
			[1,]    0    1    2    3
			[2,]    0    2    4    6
			[3,]    0    3    6    9
			[4,]    0    4    8   12
 */
arma::mat colsub(const arma::mat& D, const arma::colvec& m);


/** scale a matrix by the means of each row
 *
 *  This is comparable to the following R code
			> a=c(1,2,3,4)
			> scale(a %*% t(a))
			           [,1]       [,2]       [,3]       [,4]
			[1,] -1.1618950 -1.1618950 -1.1618950 -1.1618950
			[2,] -0.3872983 -0.3872983 -0.3872983 -0.3872983
			[3,]  0.3872983  0.3872983  0.3872983  0.3872983
			[4,]  1.1618950  1.1618950  1.1618950  1.1618950
			attr(,"scaled:center")
			[1]  2.5  5.0  7.5 10.0
			attr(,"scaled:scale")
			[1] 1.290994 2.581989 3.872983 5.163978
 *  It should be noted only the scaled:scale component is returned
 *
 */
arma::mat scale(const arma::mat& D);






/** Generate a simple matrix of zeros with val along the diagonal
 */
arma::mat diag(int val, int dim);


/** class to hold the output from the gdbr run
 *
 *  This can probably be removed, however, it mimicks the way the code from
 *  the R version (I think what looks like a component return was, in fact, a
 *  style that is used in R)
 */
struct gdbr_result {
	double num;
	double denom;
	double stat;

	gdbr_result(double num, double denom, double stat)
		: num(num), denom(denom), stat(stat) {}
};


inline gdbr_result gdbrcpp(const arma::mat& D, const arma::rowvec& y) {
	arma::mat A = -0.5 * D % D;
	int n = y.n_cols;
	arma::mat centers = scale(A);
	arma::rowvec rMeans = arma::mean(centers, 0);
	arma::mat G = colsub(centers, arma::reshape(rMeans, n, 1));
	arma::rowvec y2 = y - mean(y);

	arma::colvec yt = y2.t();
	arma::mat ones = arma::mat(1, 1, arma::fill::ones);
	arma::mat dotY2 = arma::mat(1, 1);
	dotY2.fill(arma::dot(y2, y2));
	arma::mat s=arma::solve(dotY2, ones);
	arma::mat H = vecMul(y2 * s(0), yt);
	arma::mat I = diag(1, n);
	arma::mat GHH = H * G * H;
	double num = arma::sum(GHH.diag());
	arma::mat partial_denom = ((I - H) * G * (I - H).t());
	double denom = arma::sum(partial_denom.diag());
	double stat = num / denom;

	return gdbr_result(num, denom, stat);
}




inline arma::mat vecMul(const arma::rowvec& a, const arma::colvec& b) {
	int rows = b.n_rows;
	int cols = a.n_cols;
	arma::mat results = arma::mat(rows, cols, arma::fill::zeros);
	// Fill the matrix up with copies from the column vector
	results.each_col() += b;
	// Perform the mutiplication along each row
	results.each_row() %= a;

	return results;
}

inline arma::mat colsub(const arma::mat& D, const arma::colvec& m) {
	arma::mat results(D);
	results.each_col() -= m;
	return results;
}

inline arma::mat scale(const arma::mat& D) {
	arma::rowvec means = arma::mean(D, 0);
	arma::mat results(D);
	results.each_row() -= means;
	return results;
}

inline arma::mat diag(int val, int dim) {
	arma::mat rv = arma::mat(dim, dim, arma::fill::zeros);
	for (int i=0; i<dim; i++)
		rv(i, i) = val;
	return rv;
}
