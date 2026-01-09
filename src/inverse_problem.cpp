#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "ga.h"

using std::function;
using std::vector;

// the 4th order Runge-Kutta method to solve initial value problem
vector<vector<double>> rk4(const vector<double> &x, const vector<double> &y, function<vector<double>(double, vector<double>)> f)
{
	int n_x = x.size(), n_y = y.size();

	vector<vector<double>> sol(n_x, y);
	vector<double> k1, k2, k3, k4, temp(n_y);

	int n = n_x - 1;

	for(int i = 0; i < n; i++)
	{
		double x0 = x[i], x1 = x[i + 1];
		double h = x1 - x0;
		
		k1 = f(x0, sol[i]);

		for(int j = 0; j < n_y; j++)
		{
			temp[j] = sol[i][j] + h / 2.0 * k1[j];
		}

		k2 = f(x0 + h / 2.0, temp);

		for(int j = 0; j < n_y; j++)
		{
			temp[j] = sol[i][j] + h / 2.0 * k2[j];
		}

		k3 = f(x0 + h / 2.0, temp);

		for(int j = 0; j < n_y; j++)
		{
			temp[j] = sol[i][j] + h * k3[j];
		}

		k4 = f(x1, temp);

		for(int j = 0; j < n_y; j++)
		{
			sol[i + 1][j] = sol[i][j] + h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
		}
	}

	return sol;
}

std::random_device rd;
std::mt19937 engine(rd());
std::uniform_real_distribution dist_coefs(-10.0, 10.0);

const int LEN_GENES = 3;
const int N = 20;
const double a = 1.0, b = 2.0; // bounds
vector<double> Y0{1.0, 1.0, 2.0}, X(N); // initial conditions and x-nodes
vector<vector<double>> SOLUTION(N, Y0); 

typedef ga::BaseIndividual<vector<double>, double> Individual;

Individual generator(void)
{
	vector<double> coefs(LEN_GENES);
	
	for(int i = 0; i < LEN_GENES; i++)
	{
		coefs[i] = dist_coefs(engine);
	}

	Individual I(coefs);

	return I;
}

Individual tourn_func(const vector<Individual> &population, std::function<bool(const Individual&, const Individual&)> comparator)
{
	int tourn_size = 3, pop_size = population.size();
	std::uniform_int_distribution dist(0, pop_size - 1);
	vector<Individual> temp(tourn_size);

	for(int i = 0; i < tourn_size; i++)
	{
		int k = dist(engine);
		temp[i] = population[k];
	}

	Individual best = *std::max_element(temp.begin(), temp.end(), comparator);

	return best;
}

void mate_func(Individual &i1, Individual &i2)
{
	std::uniform_int_distribution dist(0, LEN_GENES - 1);

	int k = dist(engine);

	for(int i = 0; i < k; i++)
	{
		std::swap(i1.genes[i], i2.genes[i]);
	}
}

void mutate_func(Individual &i)
{
	std::uniform_int_distribution dist(0, LEN_GENES - 1);
	double mt_range = 2.0;
	std::uniform_real_distribution mut(-mt_range, mt_range);

	int g = dist(engine);

	i.genes[g] += mut(engine);
}

/*
double func_1(double x)
{
	return 1.0 + 2.0 * x - 3.0 * x * x;
}

double func_2(double x)
{
	return -4.0 + 5.0 * x + 6.0 * x * x;
}
*/

vector<double> func(double x, vector<double> y)
{
	vector<double> v{y[1], y[2], 2.0 * y[2] - 3.0 * y[1] + 4.0 * y[0]}; 
	return v;
}

// the first three coefficients used by f1 function
double a_func(Individual &i, double x)
{
	double s = 0.0;

	for(int j = 0; j < LEN_GENES / 2; j++)
	{
		s += i.genes[j] * pow(x, j);
	}

	return s;
}

// the second three coefficients used by f2 function
double b_func(Individual &i, double x)
{
	double s = 0.0;

	for(int j = 0; j < 3; j++)
	{
		s += i.genes[j + 3] * pow(x, j);
	}
	return s;
}

vector<vector<double>> sol_func(Individual &i)
{
	double a1 = i.genes[0];
	double a2 = i.genes[1];
	double a3 = i.genes[2];

	auto f = [a1, a2, a3](double x, vector<double> y) -> vector<double>{ return {y[1], y[2], a1 * y[2] + a2 * y[1] + a3 * y[0]}; };

	vector<vector<double>> sol = rk4(X, Y0, f);

	return sol;
}

double evaluate(Individual &i)
{
	vector<vector<double>> sol = sol_func(i);

	double error = 0.0;

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < Y0.size(); j++)
		{
			double err = SOLUTION[i][j] - sol[i][j];
			error += err * err;
		}
	}

	for(double g: i.genes)
	{
		double e = std::abs(g) - 5.0;
		if(e > 0.0)
			error += e * e;
	}

	return error;
}

bool stop_cond(Individual &i)
{
	return i.fitness < 1e-5;
}

std::ostream& operator<<(std::ostream &stream, const Individual i)
{
	for(double g: i.genes)
	{
		stream << g << ' ';
	}
	stream << "  " << i.fitness;

	return stream;
}

int main(void)
{
	double h = (b - a) / (double)(N - 1);

	for(int i = 0; i < N; i++)
	{
		X[i] = a + i * h;
	}

	SOLUTION = rk4(X, Y0, func);

	double percent = 0.02;
	int max_gen = 50, pop_size = 10000, elite = (int)(pop_size * percent);
	double cxpb = 0.5, mtpb = 0.5;

	ga::GeneticAlgorithm<vector<double>, double> ga_alg(max_gen, pop_size, elite, cxpb, mtpb, generator, std::greater<Individual>(), evaluate, stop_cond, tourn_func, mate_func, mutate_func);

	Individual best = ga_alg.alg();

	std::cout << best << std::endl;

	vector<vector<double>> sol = sol_func(best);

	std::ofstream file;
	file.open("t.txt");

	for(int i = 0; i < N; i++)
	{
		file << X[i] << ' ' << sol[i][0] << ' ' << sol[i][1] << ' ' << SOLUTION[i][0] << ' ' << SOLUTION[i][1] << std::endl;
	}

	file.close();

	return 0;
}
