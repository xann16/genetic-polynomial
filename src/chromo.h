#pragma once

#ifndef ISAI_GENEPI_CHROMO_H_INCLUDED
#define ISAI_GENEPI_CHROMO_H_INCLUDED

#include "poly.h"
#include "prng.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace isai
{

  // bit masks for handling crossover
  constexpr const byte_t MSK[] = { 0xffu, 0x01u, 0x03u, 0x07u, 0x0fu,
                                   0x1fu, 0x3fu, 0x7fu, 0x0ffu };
  constexpr byte_t MASK( std::size_t pos ) noexcept { return MSK[ pos ]; }
  constexpr byte_t INV_MASK( std::size_t pos ) noexcept { return ~MSK[ pos ]; }

  // basic n-gene chromosome
  template < std::size_t N >
  class chromosome_t
  {
  public:
    // default constructor - initializes with randomized values
    chromosome_t() : m_data() { prng_t::fill_with_random_bits( m_data ); }

    // copy/move constructors/assignment
    chromosome_t( chromosome_t const & ) = default;
    chromosome_t( chromosome_t && ) noexcept = default;
    chromosome_t &operator=( chromosome_t const & ) = default;
    chromosome_t &operator=( chromosome_t && ) noexcept = default;

    // returns value of gene at given position
    bool operator[]( std::size_t pos ) const noexcept
    {
      assert( pos < gene_count() );
      return m_data[ pos >> 3u ] & ( 1u << ( pos & 0x7u ) );
    }

    // flips value of gene at given position
    void flip_gene( std::size_t pos )
    {
      assert( pos < gene_count() );
      m_data[ pos >> 3u ] ^= 1u << ( pos & 0x7u );
    }

    // proceeds with mutating this chromosome accorting to given probability
    void mutate( double mutation_rate )
    {
      assert( mutation_rate >= 0.0 && mutation_rate <= 1.0 );
      for ( auto i = std::size_t{ 0 }; i < gene_count(); i++ )
      {
        if ( prng_t::perc_check( mutation_rate ) )
        {
          flip_gene( i );
        }
      }
    }

    // crosses over this chromosome with other given one at random point
    chromosome_t crossover( chromosome_t const &other )
    {
      auto cp = prng_t::get_crossover_point< N >();
      auto cb = cp >> 3u;
      auto cboff = cp & 0x7u;
      auto res = *this;

      auto i = std::size_t{ 0 };
      while ( i < cb )
      {
        i++;
      }
      if ( cboff != 0 )
      {
        res.m_data[ i ] = ( res.m_data[ i ] & MASK( cboff ) ) |
                          ( other.m_data[ i ] & INV_MASK( cboff ) );
        i++;
      }
      while ( i < BYTE_COUNT )
      {
        res.m_data[ i ] = other.m_data[ i ];
        i++;
      }

      return res;
    }

    // number of valid genes in a chromosome
    constexpr std::size_t gene_count() const noexcept { return N; }

    // chromosome size in bytes
    static constexpr const std::size_t BYTE_COUNT =
      ( N >> 3u ) + ( ( ( N & 0x7u ) == 0 ? 0u : 1u ) );
    constexpr std::size_t size() const noexcept { return BYTE_COUNT; }

    // iterator thru chromosome bytes
    auto begin() const noexcept { return m_data.begin(); }
    auto end() const noexcept { return m_data.end(); }

  private:
    std::array< byte_t, BYTE_COUNT > m_data;
  };

  /*-------------------*/
  /*     CONVERTERS    */
  /*-------------------*/


  // converterts given chromosome to polynomial object that it represents
  template < std::size_t N >
  auto to_polynomial( chromosome_t< N > const &chromo )
  {
    auto tmp = std::array< byte_t, chromosome_t< N >::BYTE_COUNT >{};
    std::copy( chromo.begin(), chromo.end(), tmp.begin() );
    double coeffs[ N / 7u ];

    for ( auto i = std::size_t{ 0 }; i < N / 7u; i++ )
    {
      auto pos = i * 7u;
      coeffs[ i ] = 0.0;
      if ( chromo[ pos + 1 ] )
      {
        coeffs[ i ] += 8.0;
      }
      if ( chromo[ pos + 2 ] )
      {
        coeffs[ i ] += 4.0;
      }
      if ( chromo[ pos + 3 ] )
      {
        coeffs[ i ] += 2.0;
      }
      if ( chromo[ pos + 4 ] )
      {
        coeffs[ i ] += 1.0;
      }
      if ( chromo[ pos + 5 ] )
      {
        coeffs[ i ] += 0.5;
      }
      if ( chromo[ pos + 6 ] )
      {
        coeffs[ i ] += 0.25;
      }
      if ( chromo[ pos ] )
      {
        coeffs[ i ] *= -1.0;
      }
    }

    return polynomial_t< ( N / 7u ) - 1u >{ coeffs };
  }


  /*-------------------------*/
  /*     ERROR EVALUATION    */
  /*-------------------------*/

  // evaluates average linear error of polynomial represented by given
  // chromosome with respect to given training data points
  template < std::size_t N >
  double eval_error( chromosome_t< N > const &chromo,
                     training_data_t const &td )
  {
    auto poly = to_polynomial( chromo );
    auto res = double{ 0.0 };
    for ( auto &&dp : td )
    {
      auto diff = dp.y - poly( dp.x );
      // res += ( diff * diff );
      res += std::abs( diff );
    }
    return res / static_cast< double >( td.size() );
  }

}  // namespace isai

#endif  // !ISAI_GENEPI_CHROMO_H_INCLUDED
