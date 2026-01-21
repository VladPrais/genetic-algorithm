#ifndef __GA_OPERATORS_H__
#define __GA_OPERATORS_H__

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

namespace ga
{

/*
 * Selection operator
 */

template <typename Individual>
std::vector<Individual> sel_random(const std::vector<Individual> &population, size_t n, std::mt19937 &__engine__)
{
	int pop_size = population.size();
	std::vector<Individual> v(n);
	std::uniform_int_distribution<int> dist(0, pop_size - 1);

	std::generate(v.begin(), v.end(), [&](){ return population[dist(__engine__)]; });

	return v;
}

template <typename Individual>
class tournament
{
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;

	int tourn_size;

	public:

	tournament(void): tourn_size(3)
	{	}

	tournament(int tourn_size): tourn_size(tourn_size)
	{	}

	void operator()(std::vector<Individual> &gen, std::vector<Individual> &aspirants, Comparator comparator, std::mt19937 &engine)
	{
		int pop_size = gen.size();

		auto get_best = [&](){
			std::vector<Individual> rnd = sel_random(gen, tourn_size, engine);
			return *std::min_element(rnd.begin(), rnd.end(), comparator);
		};

		std::generate(aspirants.begin(), aspirants.end(), get_best);

		gen = std::move(aspirants);
	}
};

/*
 * Crossover operator
 */

template <typename Individual>
class crossover
{
	typedef std::function<void(Individual&, Individual&, std::mt19937&)> Mate_func;

	std::mt19937 engine;
	std::uniform_real_distribution<double> pb;

	double cxpb;
	Mate_func mate_func;

	public:
	
	crossover(Mate_func mate_func):
		cxpb(0.7),
		mate_func(mate_func),
		engine(rd()),
		pb(0.0, 1.0)
	{	}

	crossover(double cxpb, Mate_func mate_func):
		cxpb(cxpb),
		mate_func(mate_func),
		engine(rd()),
		pb(0.0, 1.0)
	{	}

	void operator()(std::vector<Individual> &aspirants, std::vector<Individual> child)
	{
		int pop_size = aspirants.size();

		if(pop_size % 2 != 0)
			throw std::logic_error("Population size have to be even");

		for(int i = 0; i < pop_size; i += 2)
		{
			double p = pb(engine);

			if(p < cxpb)
			{
				mate_func(aspirants[i], aspirants[i + 1], engine);

				aspirants[i].valid = false;
				aspirants[i + 1].valid = false;
			}
		}
	}
	
};

template <typename Individual>
void cx_one_point(Individual &lhs, Individual &rhs, std::mt19937 &engine)
{
	int size = std::min(lhs.genes.size(), rhs.genes.size());
	std::uniform_int_distribution<int> dist(1, size - 1);

	int point = dist(engine);

	for(int i = 0; i < point; i++)
	{
		std::swap(lhs.genes[i], rhs.genes[i]);
	}
}

/*
 * Mutation operator
 */

template <typename Individual>
class mutation
{
	typedef std::function<void(Individual&, double, std::mt19937&)> Mut_func;
	std::random_device rd;
	std::mt19937 engine;
	std::uniform_real_distribution<double> pb;

	double mtpb;
	double mgpb;
	Mut_func mut_func;

	public:

	mutation(void):
		mtpb(0.1),
		mgpb(0.05),
		engine(rd()),
		pb(0.0, 1.0)
	{	}

	mutation(double mtpb, double mgpb, Mut_func mut_func):
		mtpb(mtpb),
		mgpb(mgpb),
		engine(rd()),
		pb(0.0, 1.0),
		mut_func(mut_func)
	{	}

	void operator()(std::vector<Individual> &child)
	{
		int pop_size = child.size();

		for(int i = 0; i < pop_size; i++)
		{
			if(pb(engine) < mtpb)
			{
				mut_func(child[i], mgpb, engine);
				child[i].valid = false;
			}
		}
	}
};

template <typename Individual>
class normal_mut
{
	std::normal_distribution<double> norm;
	std::uniform_real_distribution<double> pb;

	public:

	normal_mut(void): norm(0.0, 1.0), pb(0.0, 1.0)
	{	}

	normal_mut(double mu, double sigma): norm(mu, sigma), pb(0.0, 1.0)
	{	}

	void operator()(Individual &object, double mgpb, std::mt19937 &engine)
	{
		int size = object.genes.size();

		for(int i = 0; i < size; i++)
		{
			if(pb(engine) < mgpb)
				object.genes[i] += norm(engine);
		}
	}
};

template <typename Individual>
class inverse_mut
{
	std::uniform_real_distribution<double> pb;

	public:

	inverse_mut(void): pb(0.0, 1.0)
	{	}

	void operator()(Individual &object, double mgpb, std::mt19937 &engine)
	{
		int size = object.genes.size();

		for(int i = 0; i < size; i++)
		{
			if(pb(engine) < mgpb)
				object.genes[i] = !object.genes[i];
		}
	}
};

} // namespace ga

#endif
