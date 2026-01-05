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

	void compute_fitness(std::function<FitnessType(BaseIndividual<GeneType, FitnessType> &i)> fit_func)
	{
		fitness = fit_func(*this);
		valid = true;
	}
};

template <typename GeneType, typename FitnessType>
class BaseGeneration
{
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::function<Individual(void)> Generator;
	typedef std::function<bool(Individual&, Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Eval;

	int pop_size;
	std::vector<Individual> population;
	Generator generator;
	Comparator comparator;
	Eval evaluate;

	int ind_b;
	int ind_w;

	void init(void)
	{
		std::generate(population.begin(), population.end(), generator);
	}

	public:

	BaseGeneration(int pop_size, Generator generator, Comparator comparator, Eval evaluate):
		pop_size(pop_size),
		population(pop_size),
		generator(generator),
		comparator(comparator),
		evaluate(evaluate)
	{	
		init();
	}

	int compute_fitness(void)
	{
		int evaluated_counter = 0;

		for(Individual &i: population)
		{
			if(!i.valid)
			{
				i.compute_fitness(evaluate);

				evaluated_counter++;
			}
		}
		
		auto iter_b = std::max_element(population.begin(), population.end(), comparator);
		auto iter_w = std::min_element(population.begin(), population.end(), comparator);

		ind_b = std::distance(population.begin(), iter_b);
		ind_w = std::distance(population.begin(), iter_w);

		return evaluated_counter;
	}

	Individual& get_best(void)
	{
		return population[ind_b];
	}

	Individual& get_worst(void)
	{
		return population[ind_w];
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

	int max_gen;
	int pop_size;
	double cxpb;
	double mtpb;
	int elite;

	Generator generator;
	Comparator comparator;
	Eval evaluate;
	StopCond stop_cond; // True if stop else False
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

	void config(void)
	{
		bool underflow_max_gen = max_gen < 0;
		bool underflow_pop_size = pop_size < 0;
		bool underflow_elite = elite < 0;

		if(underflow_max_gen)
		{
			throw std::underflow_error("Negative max-generations value.");
		}
		if(underflow_pop_size)
		{
			throw std::underflow_error("Negative pop-size value.");
		}
		if(underflow_elite)
		{
			throw std::underflow_error("Negative elite value.");
		}

		bool underflow_cxpb = cxpb < 0.0;
		bool underflow_mtpb = mtpb < 0.0;
		bool overflow_cxpb = cxpb > 1.0;
		bool overflow_mtpb = mtpb > 1.0;

		if(underflow_cxpb)
		{
			throw std::underflow_error("Negative crossover probability.");
		}
		if(underflow_mtpb)
		{
			throw std::underflow_error("Negative mutation probability.");
		}
		if(overflow_cxpb)
		{
			throw std::underflow_error("Overflow crossover probability.");
		}
		if(overflow_mtpb)
		{
			throw std::underflow_error("Overflow mutation probability.");
		}

		bool invalid_elite = elite >= pop_size;

		if(invalid_elite)
		{
			throw std::overflow_error("Invalid elite value: elite >= pop-size.");
		}

		bool even_elite = (elite % 2) == 0;
		bool even_pop_size = (pop_size % 2) == 0;
		bool odd_elite = !even_elite;
		bool odd_pop_size = !even_pop_size;

		if(even_pop_size and odd_elite)
		{
			throw std::logic_error("!!!");
		}
		if(odd_pop_size and even_elite)
		{
			throw std::logic_error("!!!");
		}
	}

	public:

	GeneticAlgorithm(Generator gen, Comparator comp, Eval eval, StopCond stop_cond, Selection sel_f, Crossover mate_f, Mutation mut_f):
		max_gen(50),
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
	{	
		config();
	}

	GeneticAlgorithm(int max_gen, int pop_size, double cxpb, double mtpb, int elite, Generator gen, Comparator comp, Eval eval, StopCond stop_cond, Selection sel_f, Crossover mate_f, Mutation mut_f):
		max_gen(max_gen),
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
	{	
		config();
	}

	Individual alg(void)
	{
		BaseGeneration<GeneType, FitnessType> base_gen(pop_size, generator, comparator, evaluate);
		int evaluated_counter = base_gen.compute_fitness();

		Individual best = base_gen.get_best();
		Individual worst = base_gen.get_worst();

		std::cout << "iter\tevals\tmin\tmax" << std::endl;
		std::cout << 0 << '\t' << evaluated_counter << '\t' << worst.fitness << '\t' << best.fitness << std::endl;

		bool cond = stop_cond(best);

		if(cond)
		{
			std::cout << "succes!" << std::endl;

			return best;
		}

		for(int iter = 1; iter <= max_gen; iter++)
		{
			std::vector<Individual> aspirants = sel_operator(base_gen.get());
			base_gen.set(aspirants);

			mate_operator(base_gen.get());

			mutate_operator(base_gen.get());
			
			evaluated_counter = base_gen.compute_fitness();
			best = base_gen.get_best();
			worst = base_gen.get_worst();

			std::cout << iter << '\t' << evaluated_counter << '\t' << worst.fitness << '\t' << best.fitness << std::endl;

			cond = stop_cond(best);

			if(cond)
			{
				std::cout << "succes!" << std::endl;

				return best;
			}

			/*
			for(auto g: base_gen.ptr_b->genes)
			{
				std::cout << g << ' ';
			}
			std::cout << base_gen.ptr_b->fitness << std::endl;
			*/
		}

		return best;
	}
};

} // end namespace

#endif
