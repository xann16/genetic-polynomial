#pragma once

#ifndef ISAI_GENEPI_ALG_H_INCLUDED
#define ISAI_GENEPI_ALG_H_INCLUDED

#define PRINT_EVERY 1u

#include "chromo.h"
#include "poly.h"
#include "prng.h"

#include <cmath>
#include <string>

namespace isai
{

  struct ga_settings_t
  {
    std::string batch_name = "default";
    std::vector< double > input_coeffs;

    std::size_t pop_size = 1000u;
    std::size_t max_gens = 10000u;
    std::size_t training_data_size = 50u;
    std::size_t print_interval = 1u;
    std::size_t mutation_rate_growth_threshold = 25u;
    std::size_t pop_reset_threshold = 250u;

    double error_threshold = 0.01;
    double base_mutation_rate = 0.001;
    double small_progress_rate_threshold = 0.01;
    double mutation_rate_growth_coeff = 0.5;
    double training_data_argmin = -10.0;
    double training_data_argmax = 10.0;

    bool is_input_random = true;
    bool is_verbose = false;
  };

  template < std::size_t N >
  using population_t = std::vector< chromosome_t< N > >;

  template < std::size_t N >
  class genetic_algorithm_t
  {
  public:
    // constructor - initializes all settings, population and training data
    explicit genetic_algorithm_t( ga_settings_t settings ) :
      m_settings( std::move( settings ) ),
      m_pop( m_settings.pop_size ),
      m_tdata(),
      m_fits( m_settings.pop_size ),
      m_mutation_rate( m_settings.base_mutation_rate )
    {
      assert( m_settings.is_input_random ||
              m_settings.input_coeffs.size() == 5 );
      auto poly = m_settings.is_input_random
                    ? to_polynomial( chromosome_t< 35 >{} )
                    : polynomial_t< 4 >{ m_settings.input_coeffs.data() };

      std::printf( "Initializing approximation using genetic algorithm for "
                   "polynomial: \n        " );
      poly.print();
      std::printf( "        (i.e.: P(x) =" );
      poly.print( true );
      std::printf( ")\n" );

      if ( m_settings.is_verbose )
      {
        std::printf( "Main model parameters:\n" );
        std::printf( " - population size:                %5lu\n",
                     m_settings.pop_size );
        std::printf( " - maximum generations:            %5lu\n",
                     m_settings.max_gens );
        std::printf( " - number of training data points: %5lu\n",
                     m_settings.training_data_size );
        std::printf( " - base mutation rate:             %10.4f\n",
                     m_settings.base_mutation_rate );
        std::printf( " - accepted error threshold:       %10.4f\n",
                     m_settings.error_threshold );
      }

      m_tdata = poly.get_training_data( m_settings.training_data_size,
                                        m_settings.training_data_argmin,
                                        m_settings.training_data_argmax );

      // save init data to files
      poly.to_file( std::string{ "data/" } + m_settings.batch_name +
                    "_input_poly.tsv" );
      training_data_to_file( std::string{ "data/" } + m_settings.batch_name +
                             "_training_data.tsv" );
    }

    // runs whole training process
    void run()
    {
      auto progress_file_path =
        std::string{ "data/" } + m_settings.batch_name + "_progress_data.tsv";
      auto fout =
        std::ofstream{ progress_file_path, std::ios::out | std::ios::trunc };

      m_error = 2.0 * m_settings.error_threshold;
      calculate_fitness_scores_and_error_metrics();

      while ( !check_completion_condition() )
      {
        auto temp_pop = reproduce();
        m_pop = std::move( crossover( temp_pop ) );
        mutate();
        calculate_fitness_scores_and_error_metrics();
        adjust_mutation_rate();

        // info dump
        print_progress();
        progress_to_file( fout );

        // increase generation counter
        m_curr_gen++;
      }

      print_completion_info();
    }

    // returns polynomial representing member of final population with best
    // fitness (least approx error)
    auto result() const
    {
      auto res_ch = best_individual();
      auto res = to_polynomial( res_ch );
      res.to_file( std::string{ "data/" } + m_settings.batch_name +
                   "_output_poly.tsv" );
      return std::make_pair( res, eval_error( res_ch, m_tdata ) );
    }

  private:
    /*-----------------------*/
    /*     HELPER METHODS    */
    /*-----------------------*/

    // returns index of population member with best fitness score
    std::size_t index_of_best_individual() const
    {
      auto best = std::size_t( 0 );
      for ( auto i = std::size_t{ 1 }; i < m_fits.size(); i++ )
      {
        if ( m_fits[ i ] > m_fits[ best ] )
        {
          best = i;
        }
      }
      return best;
    }

    // returns reference to chromosome with best fitness score
    auto const &best_individual() const
    {
      return m_pop[ index_of_best_individual() ];
    }

    // checks if given stopping criteria are met
    bool check_completion_condition()
    {
      return m_curr_gen >= m_settings.max_gens ||
             m_error <= m_settings.error_threshold;
    }

    /*------------------------*/
    /*     ALGORITHM STEPS    */
    /*------------------------*/

    // updates fitness scores for population
    // fitness is given by inverse of average linear error with respect to given
    // data fitness is normalized so that sum of all scores is equal to 2 * pop
    // size also updates data on current error of training data approximation
    void calculate_fitness_scores_and_error_metrics()
    {
      // calculate base fitness score: 1 / err (or big number if err == 0)
      auto index = std::size_t{ 0 };
      auto total = double{ 0.0 };
      m_avg_error = 0.0;

      for ( auto &&ch : m_pop )
      {
        auto fit = eval_error( ch, m_tdata );
        if ( fit != 0.0 )
        {
          m_avg_error += fit;
          fit = 1.0 / fit;
        }
        else
        {
          fit = 100000.0;
        }
        m_fits[ index++ ] = fit;
        total += fit;
      }

      // update population's average error
      m_avg_error /= static_cast< double >( m_pop.size() );

      // normalize fitness scores - unit = half of average fitness score
      auto half_avg_fit = total / static_cast< double >( m_pop.size() * 2u );
      for ( auto &&fit : m_fits )
      {
        fit /= half_avg_fit;
      }

      // update error of population's best member
      auto err_of_best = eval_error( best_individual(), m_tdata );

      // check if that error changed enougn - if not increment repeat counter
      auto diff = std::abs( err_of_best - m_error );
      if ( diff < m_error * m_settings.small_progress_rate_threshold )
      {
        m_best_repeats++;
      }
      m_error = err_of_best;
    }

    // creates double population by reproducing current individuals
    // proportionally to their fitness assumes proper fitness values are already
    // calculated
    population_t< N > reproduce()
    {
      auto res = population_t< N >{};
      res.reserve( m_pop.size() * 2u );

      // create reproduced individuals in proportional numbers to fitness
      auto index = std::size_t( 0 );
      for ( auto &&fit : m_fits )
      {
        auto children_count = std::floor( fit );
        for ( auto i = std::size_t{ 0 };
              i < static_cast< std::size_t >( children_count ); i++ )
        {
          res.emplace_back( m_pop[ index ] );
        }
        index++;
        fit -= children_count;
      }

      // for remaining spaces select individuals at random based on remaining
      // fractional fitness scores
      auto rem = ( m_pop.size() * 2u ) - res.size();
      auto total = std::accumulate( m_fits.begin(), m_fits.end(), 0.0 );

      auto subtotal = double{ 0.0 };
      for ( auto &&fit : m_fits )
      {
        fit /= total;
        fit += subtotal;
        subtotal = fit;
      }

      for ( auto i = std::size_t{ 0 }; i < rem; i++ )
      {
        res.emplace_back( m_pop[ prng_t::pick_by_prob( m_fits ) ] );
      }

      assert( res.size() == m_pop.size() * 2u );

      return res;
    }

    // creates new population from reproduced ones using crossover
    auto crossover( population_t< N > &parents )
    {
      prng_t::shuffle( parents );
      auto res = population_t< N >{};
      res.reserve( m_pop.size() );

      for ( auto i = std::size_t{ 0 }; i < parents.size(); i += 2 )
      {
        res.emplace_back( parents[ i ].crossover( parents[ i + 1 ] ) );
      }
      assert( m_pop.size() == res.size() );
      return res;
    }

    // applies mutations to population at given rate
    void mutate()
    {
      for ( auto &&ch : m_pop )
      {
        ch.mutate( m_mutation_rate );
      }
    }

    // adjusts actual mutation rate based on how little progress was done
    void adjust_mutation_rate()
    {
      // small no of repeats - base mutation rate
      if ( m_best_repeats < m_settings.mutation_rate_growth_threshold )
      {
        m_mutation_rate = m_settings.base_mutation_rate;
      }

      //  large number of repeats - reset population
      else if ( m_best_repeats >= m_settings.pop_reset_threshold )
      {
        if ( m_settings.is_verbose )
        {
          std::printf( "Resetting population (no significant change in best "
                       "result after %lu generations).\n",
                       m_best_repeats );
        }

        m_pop = population_t< N >( m_pop.size() );
        m_fits = std::vector< double >( m_pop.size() );
        m_mutation_rate = m_settings.base_mutation_rate;
        m_best_repeats = 0u;
        calculate_fitness_scores_and_error_metrics();
      }

      // moderate number of repeats - increase mutation rate linearly
      else
      {
        m_mutation_rate = m_settings.base_mutation_rate *
                          static_cast< double >( m_best_repeats ) *
                          m_settings.mutation_rate_growth_coeff;
      }

      if ( m_mutation_rate > 1.0 )
      {
        m_mutation_rate = 1.0;
      }
    }


    /*--------------------------*/
    /*    INFORMATION OUTPUT    */
    /*--------------------------*/

    // write generated training data points to file
    void training_data_to_file( std::string const &path )
    {
      auto fout = std::ofstream{ path, std::ios::out | std::ios::trunc };
      for ( auto &&tp : m_tdata )
      {
        fout << tp.x << '\t' << tp.y << '\n';
      }
    }

    // writes (amortized to print interval length) progress info to stdout
    void print_progress()
    {
      if ( m_settings.is_verbose )
      {
        m_error_accum += m_error;
        m_avg_error_accum += m_avg_error;

        if ( m_curr_gen % m_settings.print_interval == 0 )
        {

          std::printf( "GEN# %04lu -   avg_err: %10.3f,   best_err: %10.3f,   "
                       "reps: %7lu,   mut: %7.4f;\n",
                       m_curr_gen,
                       m_avg_error_accum / m_settings.print_interval,
                       m_error_accum / m_settings.print_interval,
                       m_best_repeats, m_mutation_rate );

          m_error_accum = 0.0;
          m_avg_error_accum = 0.0;
        }
      }
    }

    // writes progress info of every generation to file
    void progress_to_file( std::ofstream &fout )
    {
      fout << m_curr_gen << '\t' << m_error << '\t' << m_avg_error << '\t'
           << m_best_repeats << '\t' << m_mutation_rate << '\n';
    }

    // prints to stdout info abot final state
    void print_completion_info()
    {
      if ( m_settings.is_verbose )
      {
        std::printf( "\n" );
      }

      if ( m_curr_gen == m_settings.max_gens )
      {
        std::printf(
          "Training ended after reaching maximal number of generations allowed "
          "without finding solution that satisfies requested precision.\n" );
      }
      else
      {
        std::printf( "Training ended after %lu generations finding solution "
                     "that satisfies requested precision.\n",
                     m_curr_gen - 1 );
      }
    }

    // debug util printing whole pop
    void print_pop()
    {
      calculate_fitness_scores_and_error_metrics();

      auto index = std::size_t{ 0 };
      for ( auto &&ch : m_pop )
      {
        for ( auto i = std::size_t{ 0 }; i < ch.gene_count(); i++ )
        {
          if ( i % 7 == 0 )
          {
            std::printf( "|" );
          }
          std::printf( "%1d", ch[ i ] ? 1 : 0 );
        }
        std::printf( "| FIT: %10.6f ", m_fits[ index++ ] );
        std::printf( "| sqerr: %10.2f ", eval_fitness( ch, m_tdata ) );

        auto poly = isai::to_polynomial( ch );

        std::printf( "| coeffs: " );
        poly.print();
        std::printf( "\n" );
      }
    }

  private:
    ga_settings_t m_settings;

    population_t< N > m_pop;
    training_data_t m_tdata;
    std::vector< double > m_fits;

    std::size_t m_curr_gen = 1u;
    std::size_t m_best_repeats = 0u;

    double m_mutation_rate;

    double m_error = 0.0;
    double m_avg_error = 0.0;

    double m_error_accum = 0.0;
    double m_avg_error_accum = 0.0;
  };

}  // namespace isai

#endif  // !ISAI_GENEPI_ALG_H_INCLUDED
