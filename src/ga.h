#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace ga {

template <typename GeneType, typename FitnessType>
class BaseIndividual
{
	public:

	GeneType genes;
	FitnessType fitness;
	bool valid;

	BaseIndividual():
		valid(false)
	{	}

	BaseIndividual(GeneType genes):
		valid(false),
		genes(genes)
	{	}
};

template <typename GeneType, typename FitnessType>
class BaseGeneration
{
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::function<Individual(void)> Generator;
	typedef std::function<bool(Individual&, Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Eval;

	size_t pop_size;
	std::vector<Individual> population;
	Generator generator;
	Comparator comparator;
	Eval evaluate;

	void init(void)
	{
		std::generate(population.begin(), population.end(), generator);
	}

	public:

	typename std::vector<Individual>::iterator iter_b;
	typename std::vector<Individual>::iterator iter_w;

	BaseGeneration(size_t pop_size, Generator generator, Comparator comparator, Eval evaluate):
		pop_size(pop_size),
		population(pop_size),
		generator(generator),
		comparator(comparator),
		evaluate(evaluate)
	{	
		init();
	}

	size_t compute_fitness(void)
	{
		size_t evaluated_counter = 0;

		for(Individual &i: population)
		{
			if(!i.valid)
			{
				i.fitness = evaluate(i);
				i.valid = true;

				evaluated_counter++;
			}
		}
		
		iter_b = std::max_element(population.begin(), population.end(), comparator);
		iter_w = std::min_element(population.begin(), population.end(), comparator);

		return evaluated_counter;
	}

	void set(std::vector<Individual> &pop)
	{
		population = pop;
	}

	std::vector<Individual>& get(void)
	{
		return population;
	}

	void sort(void)
	{
		std::sort(population.begin(), population.end(), comparator);
	}

	Individual& operator[](size_t index)
	{
		return population[index];
	}
};

template <typename GeneType, typename FitnessType>
class GeneticAlgorithm
{	
	typedef BaseIndividual<GeneType, FitnessType> Individual;

	typedef std::function<Individual(void)> Generator;
	typedef std::function<bool(Individual&, Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Eval;
	typedef std::function<bool(Individual&)> StopCond;

	typedef std::function<Individual(std::vector<Individual>&)> Selection;
	typedef std::function<void(Individual&, Individual&)> Crossover;
	typedef std::function<void(Individual&)> Mutation;

	std::random_device rd;
	std::mt19937 engine;
	std::uniform_real_distribution<double> pb;
	std::uniform_int_distribution<int> dist;

	size_t max_generations;
	size_t pop_size;
	double cxpb;
	double mtpb;
	size_t elite;
	//std::vector<double> params;

	Generator generator;
	Comparator comparator;
	Eval evaluate;
	StopCond stop_cond; // 1 if success else 0
	Selection sel_func;
	Crossover mate_func;
	Mutation mutate_func;

	std::vector<Individual> sel_operator(std::vector<Individual> &population)
	{
		std::vector<Individual> aspirants(population);

		std::sort(aspirants.begin(), aspirants.end(), comparator);

		int c = pop_size - 1;

		for(int i = 0; i < elite; i++)
		{
			aspirants[i] = aspirants[c];
			c--;
		}

		for(int i = elite; i < pop_size; i++)
		{
			aspirants[i] = sel_func(population);
		}

		return aspirants;
	}

	void mate_operator(std::vector<Individual> &aspirants)
	{
		int range = pop_size - 1;

		for(int i = 0; i < range; i += 2)
		{
			double p = pb(engine);

			if(p < cxpb)
			{
				mate_func(aspirants[i], aspirants[i + 1]);

				aspirants[i].valid = false;
				aspirants[i + 1].valid = false;
			}
		}
	}

	void mutate_operator(std::vector<Individual> &aspirants)
	{
		for(int i = 0; i < pop_size; i++)
		{
			double p = pb(engine);

			if(p < mtpb)
			{
				mutate_func(aspirants[i]);

				aspirants[i].valid = false;
			}
		}
	}

	public:

	GeneticAlgorithm(Generator gen, Comparator comp, Eval eval, StopCond stop_cond, Selection sel_f, Crossover mate_f, Mutation mut_f):
		max_generations(50),
		pop_size(100),
		cxpb(0.7),
		mtpb(0.1),
		elite(5),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop_cond),
		sel_func(sel_f),
		mate_func(mate_f),
		mutate_func(mut_f),
		engine(rd()),
		pb(0.0, 1.0),
		dist(0, pop_size - 1)
	{	}

	GeneticAlgorithm(size_t max_gen, size_t pop_size, double cxpb, double mtpb, size_t elite, Generator gen, Comparator comp, Eval eval, StopCond stop_cond, Selection sel_f, Crossover mate_f, Mutation mut_f):
		max_generations(max_gen),
		pop_size(pop_size),
		cxpb(cxpb),
		mtpb(mtpb),
		elite(elite),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop_cond),
		sel_func(sel_f),
		mate_func(mate_f),
		mutate_func(mut_f),
		engine(rd()),
		pb(0.0, 1.0),
		dist(0, pop_size - 1)
	{	}

	Individual alg(void)
	{
		BaseGeneration<GeneType, FitnessType> base_gen(pop_size, generator, comparator, evaluate);
		size_t evaluated_counter = base_gen.compute_fitness();

		std::cout << "iter\tevals\tmin\tmax" << std::endl;

		std::cout << 0 << '\t' << evaluated_counter << '\t' << base_gen.iter_w->fitness << '\t' << base_gen.iter_b->fitness << std::endl;

		bool cond = stop_cond(*base_gen.iter_b);

		if(cond)
		{
			std::cout << "succes!" << std::endl;

			return *base_gen.iter_b;
		}

		for(size_t iter = 1; iter <= max_generations; iter++)
		{
			std::vector<Individual> aspirants = sel_operator(base_gen.get());
			base_gen.set(aspirants);

			mate_operator(base_gen.get());

			mutate_operator(base_gen.get());
			
			evaluated_counter = base_gen.compute_fitness();

			std::cout << iter << '\t' << evaluated_counter << '\t' << base_gen.iter_w->fitness << '\t' << base_gen.iter_b->fitness << std::endl;

			cond = stop_cond(*base_gen.iter_b);

			if(cond)
			{
				std::cout << "succes!" << std::endl;

				return *base_gen.iter_b;
			}
		}

		return *base_gen.iter_b;
	}
};

} // end namespace

#endif
