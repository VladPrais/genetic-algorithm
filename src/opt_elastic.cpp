#include <array>
#include <functional>
#include <iostream>
#include <vector>
#include <math.h>

using std::array;
using std::vector;
using std::function;

class ODE
{
	private:

	function<vector<double>(double, vector<double>, vector<double>)> eq;
	vector<double> init_conds;
	vector<double> domain;

	public:

	ODE(function<vector<double>(double, vector<double>, vector<double>)> eq, vector<double> init_conds, vector<double> domain):
		eq(eq),
		init_conds(init_conds),
		domain(domain){	}

	vector<vector<double>> rk4(vector<double> params)
	{
		if(eq(domain[0], init_conds, params).size() != init_conds.size())
		{
			throw std::runtime_error("fuck it");
		}

		int x_size = domain.size(), ic_size = init_conds.size();
		vector<double> k1, k2, k3, k4;
		vector<double> y_temp(init_conds);
		double h, x0, x1;

		vector<vector<double>> sol(x_size, init_conds);


		for(int i = 0; i < x_size - 1; i++)
		{
			x0 = domain[i];
			x1 = domain[i + 1];
			h = x1 - x0;
			k1 = eq(x0, sol[i], params);

			for(int j = 0; j < ic_size; j++)
			{
				y_temp[j] = sol[i][j] + h / 2.0 * k1[j];
			}
			k2 = eq(x0 + h / 2.0, y_temp, params);

			for(int j = 0; j < ic_size; j++)
			{
				y_temp[j] = sol[i][j] + h / 2.0 * k2[j];
			}
			k3 = eq(x0 + h / 2.0, y_temp, params);

			for(int j = 0; j < ic_size; j++)
			{
				y_temp[j] = sol[i][j] + h * k3[j];
			}
			k4 = eq(x0 + h, y_temp, params);

			for(int j = 0; j < ic_size; j++)
			{
				sol[i + 1][j] = sol[i][j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
			}
		}

		return sol;
	}
}; 

double g(double x)
{
	return 5;
}
double r(double x)
{
	return 1;
}

/*
 * (g(x)u`(x))` + Lr(x)u(x) = 0, u(0) = 0, g(1)u`(1) = 1
 *
 * g(x)u`(x) = t
 * t` = -Lr(x)u(x)
 */

vector<double> dy(double x, vector<double> y, vector<double> params)
{
	double L = params[0];
	//vector<double> v{y[1], -L * r(x) * y[0]};
	vector<double> v{y[1] / g(x), -L * r(x) * y[0]};
	return v;
}

/*
vector<double> dy(double x, vector<double> y, vector<double> params)
{
	vector<double> v{y[1], 6*x + 9*x*x + x*x*x - y[0] - 3*y[1]};
	return v;
}
*/

double fpm(function<double(double, vector<double>)> f, double x0, double x1, vector<double> p)
{
	double x2, eps = 1e-6;
	int max_step = (int)1e5;

	for(int i = 0; i < max_step; i++)
	{
		x2 = (f(x1, p) * x0 - f(x0, p) * x1) / (f(x1, p) - f(x0, p));

		if(abs(x2 - x1) < eps)
		{
			break;
		}

		x0 = x1;
		x1 = x2;
	}
	return x1;
}

double sh(double C, vector<double> p)
{
	int n = 50;
	vector<double> x(n);
	double h = 1.0 / (double)(n - 1);

	for(int i = 0; i < n; i++)
	{
		x[i] = i * h;
	}

	vector<double> y0{0, C};
	ODE ode(dy, y0, x);
	double target = ode.rk4(p)[n-1][1];

	return abs(target / g(1.0) - 1.0);
}

vector<double> achh(vector<double> &x, int k, double h)
{
	vector<double> p(1);
	vector<vector<double>> sol;
	vector<double> al;
	vector<double> y0(2, 0);

	for(int i = 0; i < k; i++)
	{
		p[0] = i * h;
		double a = fpm(sh, 0, 1, p);
		y0[1] = a;
		ODE ode(dy, y0, x);
		sol = ode.rk4(p);
		double val = sol[sol.size() - 1][0];
		std::cout << i * h << '\t' << val << std::endl;
		al.push_back(val);
	}

	return al;
}


int main(void)
{
	int n = 15;
	vector<double> x(n);
	double h = 1.0 / (double)(n - 1);
	for(int i = 0; i < x.size(); i++)
	{
		x[i] += h * i;
	}

	vector<double> p{2};
	double a = fpm(sh, 0, 1, p);

	std::cout << "a = " << a << std::endl;
	vector<double> y0{0.0, a};

	ODE ode2(dy, y0, x);
	vector<vector<double>> sol;
	sol = ode2.rk4(p);

	/*
	for(int i = 0; i < n; i++)
	{
		std::cout << x[i] << '\t';
		for(double j:sol[i])
		{
			std::cout << j << '\t';
		}
		std::cout << std::endl;
	}
	*/

	achh(x, 100, 0.5);

	return 0;
}
