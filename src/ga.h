#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "operators_ga.h"

namespace ga {

template <typename GeneType, typename FitnessType>
struct BaseIndividual
{
	GeneType genes;
	FitnessType fitness;
	bool valid;

	BaseIndividual():
		valid(false)
	{	}

	BaseIndividual(GeneType &genes):
		valid(false),
		genes(genes)
	{	}

	void compute_fitness(std::function<FitnessType(BaseIndividual<GeneType, FitnessType> &i)> fit_func)
	{
		fitness = fit_func(*this);
		valid = true;
	}

	bool is_valid(void)
	{
		return valid;
	}
};

template <typename GeneType, typename FitnessType>
struct BaseGenetic
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Changer =  std::function<void(Population&, std::mt19937&)>;

	std::random_device rd;
	std::mt19937 engine;

	int max_gen;
	int pop_size;
	int elite;

	Generator generator;
	Comparator comparator;
	Evaluate evaluate;
	StopCond stopcond;

	Selection sel;
	Changer mate;
	Changer mut;

	BaseGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		max_gen(50),
		pop_size(100),
		elite(10),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stopcond(stop),
		sel(sel),
		mate(mate),
		mut(mut),
		engine(rd())
	{
		config();
	}

	BaseGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		max_gen(max_gen),
		pop_size(pop_size),
		elite(elite),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stopcond(stop),
		sel(sel),
		mate(mate),
		mut(mut),
		engine(rd())
	{
		config();
	}

	void config(void)
	{
		bool underflow_max_gen = max_gen <= 0;
		bool underflow_pop_size = pop_size <= 0;
		bool underflow_elite = elite < 0;
		bool overflow_elite = elite >= pop_size;

		if(underflow_max_gen)
			throw std::underflow_error("Underflow max generations value " + std::to_string(max_gen));
		if(underflow_pop_size)
			throw std::underflow_error("Underflow population size value " + std::to_string(pop_size));
		if(underflow_elite)
			throw std::underflow_error("Underflow elite value " + std::to_string(elite));
		if(overflow_elite)
			throw std::underflow_error("Overflow elite value " + std::to_string(elite));
	}
};

template <typename GeneType, typename FitnessType>
class GeneticAlgorithm: private BaseGenetic<GeneType, FitnessType>
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Changer =  std::function<void(Population&, std::mt19937&)>;

	/*
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::vector<Individual> Population;

	typedef std::function<Individual(void)> Generator;
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Eval;
	typedef std::function<bool(Individual&)> StopCond;

	typedef std::function<void(Population&, Comparator, std::mt19937&)> Selection;
	typedef std::function<void(Population&, std::mt19937&)> Changer;
	*/


	/*
	int max_gen;
	int pop_size;
	int elite;
	*/

	Population population;

	/*
	Generator generator;
	Comparator comparator;
	Eval evaluate;
	StopCond stop_cond; // True if stop else False
	
	Selection sel_operator;
	Changer mate_operator;
	Changer mut_operator;

	Population population;
	*/

	public:

	GeneticAlgorithm(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut),
		/*
		max_gen(50),
		pop_size(100),
		elite(5),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop_cond),
		sel_operator(sel_operator),
		mate_operator(mate_operator),
		mut_operator(mut_operator),
		*/
		population(this->pop_size)
	{	
	}

	GeneticAlgorithm(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		BaseGenetic<GeneType, FitnessType>(max_gen, pop_size, elite, gen, comp, eval, stop, sel, mate, mut),
		/*
		max_gen(max_gen),
		pop_size(pop_size),
		elite(elite),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop_cond),
		sel_operator(sel_operator),
		mate_operator(mate_operator),
		mut_operator(mut_operator),
		engine(rd()),
		pb(0.0, 1.0),
		dist(0, pop_size - 1),
		*/
		population(this->pop_size)
	{	
	}

	Individual operator()(void)
	{
		std::generate(population.begin(), population.end(), this->generator);
		int evaluated_counter = compute_fitness(population);

		Individual best = get_best(population);
		Individual worst = get_worst(population);

		auto ch = '\t';

		std::cout << "iter" << ch << "evals" << ch << "low-fit" << ch << "high-fit" << std::endl;
		std::cout << 0 << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << std::endl;

		if(this->stopcond(best))
		{
			std::cout << "Success!" << std::endl;
			return best;
		}

		for(int iter = 1; iter <= this->max_gen; iter++)
		{
			evaluated_counter = one_iter(population);

			best = get_best(population);
			worst = get_worst(population);

			std::cout << iter << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << ch << std::endl;

			if(this->stopcond(best))
			{
				std::cout << "Success!" << std::endl;
				return best;
			}
		}	

		return best;
	}

	private:

	int one_iter(Population &pop)
	{
		this->sel(pop, this->comparator, this->engine);
		this->mate(pop, this->engine);
		this->mut(pop, this->engine);

		return compute_fitness(population);
	}

	int compute_fitness(Population &population)
	{
		int evaluated_counter = 0;

		for(Individual &i: population)
		{
			if(!i.is_valid())
			{
				i.compute_fitness(this->evaluate);

				evaluated_counter++;
			}
		}
		return evaluated_counter;
	}

	Individual get_best(Population &population) const
	{
		auto iter_best = std::min_element(population.begin(), population.end(), this->comparator);

		return *iter_best;
	}

	Individual get_worst(Population &population) const
	{
		auto iter_worst = std::max_element(population.begin(), population.end(), this->comparator);

		return *iter_worst;
	}

	/*
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
	*/
};

} // namespace ga

#endif
