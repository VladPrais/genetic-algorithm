#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "operators_ga.h"

namespace ga {

template <typename GeneType, typename FitnessType>
struct BaseIndividual
{
	std::vector<GeneType> genes;
	FitnessType fitness;
	bool valid;

	BaseIndividual():
		valid(false)
	{	}

	BaseIndividual(std::vector<GeneType> &genes):
		valid(false),
		genes(genes)
	{	}

	void compute_fitness(std::function<FitnessType(BaseIndividual<GeneType, FitnessType> &i)> fit_func)
	{
		fitness = fit_func(*this);
		valid = true;
	}

	GeneType& operator[](size_t n)
	{
		return genes[n];
	}

	const GeneType& operator[](size_t n) const
	{
		return genes[n];
	}
};

template <typename GeneType, typename FitnessType>
class BaseGeneration
{
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::function<Individual(void)> Generator;
	typedef std::function<FitnessType(Individual&)> Eval;

	int pop_size;
	Generator generator;
	Eval evaluate;

	void init(void)
	{
		std::generate(population.begin(), population.end(), generator);
	}

	public:

	std::vector<Individual> population;

	BaseGeneration(int pop_size, Generator generator, Eval evaluate):
		pop_size(pop_size),
		population(pop_size),
		generator(generator),
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
		return evaluated_counter;
	}

	Individual get_best(std::function<bool(const Individual&, const Individual&)> comparator)
	{
		auto iter_best = std::min_element(population.begin(), population.end(), comparator);

		return *iter_best;
	}

	Individual get_worst(std::function<bool(const Individual&, const Individual&)> comparator)
	{
		auto iter_worst = std::max_element(population.begin(), population.end(), comparator);

		return *iter_worst;
	}

	Individual& operator[](size_t n)
	{
		population[n].valid = false;
		return population[n];
	}

	const Individual& operator[](size_t n) const
	{
		return population[n];
	}
};

template <typename GeneType, typename FitnessType>
class GeneticAlgorithm
{	
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::vector<Individual> Population;

	typedef std::function<Individual(void)> Generator;
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Eval;
	typedef std::function<bool(Individual&)> StopCond;

	typedef std::function<void(Population&, Comparator, std::mt19937&)> Selection;
	typedef std::function<void(Population&, std::mt19937&)> Changer;

	std::random_device rd;
	std::mt19937 engine;
	std::uniform_real_distribution<double> pb;
	std::uniform_int_distribution<int> dist;

	int max_gen;
	int pop_size;
	int elite;
	/*
	double cxpb;
	double mtpb;
	double mgpb;
	*/

	Generator generator;
	Comparator comparator;
	Eval evaluate;
	StopCond stop_cond; // True if stop else False
			    //
	Selection sel_operator;
	Changer mate_operator;
	Changer mut_operator;

	public:

	GeneticAlgorithm(Generator gen, Comparator comp, Eval eval, StopCond stop_cond, Selection sel_operator, Changer mate_operator, Changer mut_operator):
		max_gen(50),
		pop_size(100),
		elite(5),
		/*
		cxpb(0.7),
		mtpb(0.1),
		mgpb(0.05),
		*/
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop_cond),
		sel_operator(sel_operator),
		mate_operator(mate_operator),
		mut_operator(mut_operator),
		engine(rd()),
		pb(0.0, 1.0),
		dist(0, pop_size - 1)
	{	
		config();
	}

	GeneticAlgorithm(int max_gen, int pop_size, int elite, /*double cxpb, double mtpb, double mgpb,*/ Generator gen, Comparator comp, Eval eval, StopCond stop_cond, Selection sel_operator, Changer mate_operator, Changer mut_operator):
		max_gen(max_gen),
		pop_size(pop_size),
		elite(elite),
		/*
		cxpb(cxpb),
		mtpb(mtpb),
		mgpb(mgpb),
		*/
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop_cond),
		sel_operator(sel_operator),
		mate_operator(mate_operator),
		mut_operator(mut_operator),
		engine(rd()),
		pb(0.0, 1.0),
		dist(0, pop_size - 1)
	{	
		config();
	}

	Individual operator()(void)
	{
		BaseGeneration<GeneType, FitnessType> base_gen(pop_size, generator, evaluate);
		int evaluated_counter = base_gen.compute_fitness();

		Individual best = base_gen.get_best(comparator);
		Individual worst = base_gen.get_worst(comparator);

		auto ch = '\t';

		std::cout << "iter" << ch << "evals" << ch << "low-fit" << ch << "high-fit" << std::endl;
		std::cout << 0 << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << std::endl;

		if(stop_cond(best))
		{
			std::cout << "Success!" << std::endl;
			return best;
		}

		for(int iter = 1; iter <= max_gen; iter++)
		{
			sel_operator(base_gen.population, comparator, engine);
			mate_operator(base_gen.population, engine);
			mut_operator(base_gen.population, engine);

			evaluated_counter = base_gen.compute_fitness();

			best = base_gen.get_best(comparator);
			worst = base_gen.get_worst(comparator);

			std::cout << iter << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << ch << std::endl;

			if(stop_cond(best))
			{
				std::cout << "Success!" << std::endl;
				return best;
			}
		}	

		return best;
	}

	private:

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

		/*
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
		*/

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
};

} // namespace ga

#endif
