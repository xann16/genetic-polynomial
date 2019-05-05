#pragma once

#ifndef ISAI_GENEPI_ALG_H_INCLUDED
#define ISAI_GENEPI_ALG_H_INCLUDED

#define PRINT_EVERY 1u

#include "chromo.h"
#include "poly.h"
#include "prng.h"

#include <cmath>

namespace isai
{
  template < std::size_t N >
  using population_t = std::vector< chromosome_t< N > >;

  template < std::size_t N >
  class genetic_algorithm_t
  {
  public:
    explicit genetic_algorithm_t( std::size_t pop_size, training_data_t td ) :
      m_pop( pop_size ),
      m_tdata( std::move( td ) ),
      m_fits( pop_size )
    {
      assert( pop_size % 2 == 0 );
    }
    explicit genetic_algorithm_t( std::size_t pop_size, training_data_t td,
                                  std::size_t max_gen, double mutation_rate,
                                  double max_sqerr ) :
      m_pop( pop_size ),
      m_tdata( std::move( td ) ),
      m_fits( pop_size ),
      m_max_generations( max_gen ),
      m_mutation_rate_base( mutation_rate ),
      m_mutation_rate( mutation_rate ),
      m_max_sqerror( max_sqerr )
    {
      assert( pop_size % 2 == 0 );
    }

    /*
    void train()
    {
      // initialize (pop already present)
      update_fitness();

      for ( auto gen = std::size_t{ 1 }; gen <= m_max_generations; gen++ )
      {
        auto new_pop = population_t< N >{};
        new_pop.reserve( m_pop.size() );
        for ( auto pair_index = std::size_t{ 0 };
              pair_index < m_pop.size() / 2u; pair_index++ )
        {
          auto mommy_index = prng_t::pick_by_prob( m_fits );
          auto daddy_index = prng_t::pick_by_prob( m_fits );

          auto mommy = m_pop[ mommy_index ];
          auto daddy = m_pop[ daddy_index ];

          if ( prng_t::perc_check( m_crossover_rate ) )
          {
            auto [ child1, child2 ] = mommy.crossover( daddy );
            new_pop.emplace_back( child1.mutated( m_mutation_rate ) );
            new_pop.emplace_back( child2.mutated( m_mutation_rate ) );
          }
          else
          {
            new_pop.emplace_back( mommy.mutated( m_mutation_rate ) );
            new_pop.emplace_back( daddy.mutated( m_mutation_rate ) );
          }
        }

        assert( new_pop.size() == m_pop.size() );
        m_pop = std::move( new_pop );

        update_fitness();

        if ( gen % PRINT_EVERY == 0 )
        {
          m_curr_error_backlog += m_curr_error;
          std::printf( "Generation %6lu... (error: %10.2f)\n", gen,
                       m_curr_error_backlog / PRINT_EVERY );
          m_curr_error_backlog = 0.0;
        }
        else
        {
          m_curr_error_backlog += m_curr_error;
        }
      }
      std::puts( "Evolution completed" );
    }
    */

    population_t< N > reproduce()
    {
      auto res = population_t< N >{};
      res.reserve( m_pop.size() * 2u );

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

    void mutate()
    {
      for ( auto &&ch : m_pop )
      {
        ch.mutate( m_mutation_rate );
      }
    }

    void train()
    {
      update_fitness();
      m_curr_best_error = m_max_sqerror * 2.0;

      for ( auto gen = std::size_t{ 1 }; !is_done( gen ); gen++ )
      {
        auto temp_pop = reproduce();
        m_pop = std::move( crossover( temp_pop ) );
        mutate();
        update_fitness();
        // print_pop();

        calc_best_error();

        if ( gen % PRINT_EVERY == 0 )
        {
          m_curr_error_backlog += m_curr_error;
          m_curr_best_error_backlog += m_curr_best_error;
          std::printf( "GEN# %04lu:   avg_err: %10.3f,   best_err: %8.3f, "
                       "reps:   %3lu, mut:   %5.3f;\n",
                       gen, m_curr_error_backlog / PRINT_EVERY,
                       m_curr_best_error_backlog / PRINT_EVERY, m_best_repeats,
                       m_mutation_rate );
          m_curr_error_backlog = 0.0;
          m_curr_best_error_backlog = 0.0;
        }
        else
        {
          m_curr_error_backlog += m_curr_error;
          m_curr_best_error_backlog += m_curr_best_error;
        }

        adjust_mutation_rate();
      }
      std::puts( "Evolution completed" );
    }


    auto get_winner()
    {
      update_fitness();
      return m_pop[ get_best() ];
    }

    std::size_t get_best()
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

    void calc_best_error()
    {
      auto best_err = eval_fitness( m_pop[ get_best() ], m_tdata );
      auto diff = std::abs( best_err - m_curr_best_error );
      if ( diff < m_curr_best_error * 0.01 )
      {
        m_best_repeats++;
      }
      // else if ( diff > m_curr_best_error * 0.25 )
      //{
      //  m_best_repeats = 0u;
      //}
      m_curr_best_error = best_err;
    }

    bool is_done( std::size_t curr_gen )
    {
      return curr_gen >= m_max_generations ||
             m_curr_best_error <= m_max_sqerror;
    }

    void adjust_mutation_rate()
    {
      if ( m_best_repeats < 25u )
      {
        m_mutation_rate = m_mutation_rate_base;
      }
      else if ( m_best_repeats >= 250u )
      {
        std::printf( "Resetting population (no change in best result after %lu "
                     "generations)...\n",
                     m_best_repeats );
        m_pop = population_t< N >( m_pop.size() );
        m_fits = std::vector< double >( m_pop.size() );
        m_mutation_rate = m_mutation_rate_base;
        m_best_repeats = 0u;
        update_fitness();
      }
      /*
      else if ( m_best_repeats >= 100u )
      {
        m_mutation_rate = m_mutation_rate_base * 100.0;
      }
      else if ( m_best_repeats >= 50u )
      {
        m_mutation_rate = m_mutation_rate_base * 50.0;
      }
      else
      {
        m_mutation_rate = m_mutation_rate_base * 10.0;
      }
      */
      else
      {
        m_mutation_rate =
          m_mutation_rate_base * static_cast< double >( m_best_repeats ) * 0.5;
      }
    }

    void print_pop()
    {
      update_fitness();

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

        std::printf( "| coeffs: [ " );
        for ( auto i = std::size_t{ 0 }; i < poly.size(); i++ )
        {
          std::printf( "%5.2f ", poly[ i ] );
        }
        std::printf( "]\n" );
      }
    }

  private:
    /*
    void update_fitness()
    {
      auto index = std::size_t{ 0 };
      auto total = double{ 0.0 };
      m_curr_error = 0.0;
      for ( auto &&ch : m_pop )
      {
        auto fit = eval_fitness( ch, m_tdata );
        m_curr_error += fit;
        fit = 1.0 / fit;
        m_fits[ index++ ] = fit;
        total += fit;
      }

      auto subtot = double{ 0.0 };
      for ( auto &&f : m_fits )
      {
        f /= total;
        f += subtot;
        subtot = f;
      }

      m_curr_error /= static_cast< double >( m_pop.size() );
    }
    */

    void update_fitness()
    {
      auto index = std::size_t{ 0 };
      auto total = double{ 0.0 };
      m_curr_error = 0.0;
      for ( auto &&ch : m_pop )
      {
        auto fit = eval_fitness( ch, m_tdata );
        if ( fit != 0.0 )
        {
          m_curr_error += fit;
          fit = 1.0 / fit;
        }
        else
        {
          fit = 100000.0;
        }
        m_fits[ index++ ] = fit;
        total += fit;
      }

      m_curr_error /= static_cast< double >( m_pop.size() );
      auto half_avg_fit = total / static_cast< double >( m_pop.size() * 2u );

      for ( auto &&fit : m_fits )
      {
        fit /= half_avg_fit;
      }
    }


  private:
    population_t< N > m_pop;
    training_data_t m_tdata;
    std::vector< double > m_fits;

    std::size_t m_best_repeats = 0u;

    std::size_t m_max_generations = 1000u;
    double m_mutation_rate_base = 0.001;
    double m_mutation_rate = 0.001;
    double m_max_sqerror = 1.0;

    double m_curr_error = 0.0;
    double m_curr_error_backlog = 0.0;

    double m_curr_best_error = 0.0;
    double m_curr_best_error_backlog = 0.0;
  };  // namespace isai

}  // namespace isai

#endif  // !ISAI_GENEPI_ALG_H_INCLUDED
