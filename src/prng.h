#pragma once

#ifndef ISAI_GENEPI_PRNG_H_INCLUDED
#define ISAI_GENEPI_PRNG_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <vector>

namespace isai
{

  using byte_t = unsigned char;

  class prng_t
  {
  private:
    prng_t() noexcept = default;

  public:
    static void initialize() noexcept
    {
      s_eng = std::default_random_engine{ s_dev() };
    }

    static std::vector< double >
    get_uniform_doubles( std::size_t count, double lo, double hi ) noexcept
    {
      auto dist = std::uniform_real_distribution< double >{ lo, hi };
      auto res = std::vector< double >{};
      res.reserve( count );

      for ( auto i = std::size_t{ 0 }; i < count; i++ )
      {
        res.emplace_back( dist( s_eng ) );
      }

      return res;
    }

    template < std::size_t N >
    static void fill_with_random_bits( std::array< byte_t, N > &bit_array )
    {
      auto dist = std::uniform_int_distribution< byte_t >{};
      for ( auto &b : bit_array )
      {
        b = dist( s_eng );
      }
    }

    static bool perc_check( double perc ) noexcept
    {
      assert( perc >= 0.0 );
      assert( perc <= 1.0 );
      return perc >= std::generate_canonical< double, 64 >( s_eng );
    }

    template < std::size_t N >
    static std::size_t get_crossover_point()
    {
      auto dist = std::uniform_int_distribution< std::size_t >{ 1, N };
      return dist( s_eng );
    }

    static std::size_t pick_by_prob( std::vector< double > const &table )
    {
      auto val = std::generate_canonical< double, 64 >( s_eng );
      for ( auto i = std::size_t{ 0 }; i < table.size(); i++ )
      {
        if ( val <= table[ i ] )
        {
          return i;
        }
      }
      return table.size() - 1;
    }

    template < typename T >
    static void shuffle( std::vector< T > &v )
    {
      std::random_shuffle( std::begin( v ), std::end( v ) );
    }

  private:
    static std::random_device s_dev;
    static std::default_random_engine s_eng;
  };

}  // namespace isai

#endif  // !ISAI_GENEPI_POLY_H_INCLUDED
