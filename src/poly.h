#pragma once

#ifndef ISAI_GENEPI_POLY_H_INCLUDED
#define ISAI_GENEPI_POLY_H_INCLUDED

#include "prng.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace isai
{

  // custom struct storing single training data point
  struct data_point_t
  {
    double x;
    double y;
  };

  // type alias for array of training data points
  using training_data_t = std::vector< data_point_t >;


  // rsimple representation of arbitrary polynomial of order N
  template < std::size_t N >
  class polynomial_t
  {
  public:
    // default constructor - zero polynomial
    polynomial_t() noexcept : m_data()
    {
      std::memset( m_data.data(), 0, size() * sizeof( double ) );
    }

    // default copy/move constructors/assignments
    polynomial_t( polynomial_t const & ) = default;
    polynomial_t( polynomial_t && ) noexcept = default;
    polynomial_t &operator=( polynomial_t const & ) = default;
    polynomial_t &operator=( polynomial_t && ) noexcept = default;
    ~polynomial_t() noexcept = default;

    // constructor from initializer list
    polynomial_t( std::initializer_list< double > il ) noexcept : m_data()
    {
      assert( il.size() == size() );
      auto index = std::size_t{ 0 };
      for ( auto &&val : il )
      {
        m_data[ index++ ] = val;
      }
    }

    // constructor from raw array of coefficients
    explicit polynomial_t( double *vals_p ) noexcept : m_data()
    {
      std::memcpy( m_data.data(), vals_p, size() * sizeof( double ) );
    }

    // accesses a_index coefficient
    double operator[]( std::size_t index ) const noexcept
    {
      assert( index < size() );
      return m_data[ index ];
    }

    // accesses a_index coefficient
    double &operator[]( std::size_t index ) noexcept
    {
      assert( index < size() );
      return m_data[ index ];
    }

    // evaluates value of this polinomial at given argument
    double operator()( double arg ) const noexcept
    {
      auto res = m_data[ order() ];
      for ( auto i = static_cast< int >( order() ) - 1; i >= 0; i-- )
      {
        res *= arg;
        res += m_data[ static_cast< std::size_t >( i ) ];
      }
      return res;
    }

    // number of coefficients describing polynomial (order + 1)
    constexpr std::size_t size() const noexcept { return N + 1; }

    // order of given polynomial
    constexpr std::size_t order() const noexcept { return N; }

    // generates random training set for this polynomial
    training_data_t get_training_data( std::size_t count, double argmin = -10.0,
                                       double argmax = 10.0 )
    {
      auto args = prng_t::get_uniform_doubles( count, argmin, argmax );
      auto res = training_data_t{};
      res.reserve( count );

      std::transform( std::begin( args ), std::end( args ),
                      std::back_inserter( res ), [this]( double arg ) {
                        return data_point_t{ arg, ( *this )( arg ) };
                      } );

      return res;
    }

    // prints polynomial to stdout - either as coefficients vector or in fancy,
    // human-readable form
    void print( bool is_fancy = false ) const
    {
      if ( is_fancy )
      {
        bool is_started = false;
        for ( auto i = static_cast< int >( order() ); i >= 0; i-- )
        {
          auto val = m_data[ i ];

          // sign
          if ( val < 0.0 )
          {
            std::printf( " - " );
          }
          else if ( val > 0.0 )
          {
            if ( is_started )
            {
              std::printf( " + " );
            }
            else
            {
              std::printf( " " );
            }
          }
          else
          {
            continue;
          }

          // coeff
          std::printf( "%.2f", std::abs( val ) );

          // variable
          if ( i != 0 )
          {
            std::printf( "x^%d", i );
          }
        }
      }
      else
      {
        std::printf( "[ " );
        for ( auto i = std::size_t{ 0 }; i < size(); i++ )
        {
          std::printf( "%.2f ", m_data[ i ] );
        }
        std::printf( "]\n" );
      }
    }

    // writes given polynomial (its coefficients) to file
    void to_file( std::string const &path )
    {
      auto fout = std::ofstream{ path, std::ios::out | std::ios::trunc };
      for ( auto &&coeff : m_data )
      {
        fout << coeff << '\n';
      }
    }

  private:
    std::array< double, N + 1 > m_data;
  };

}  // namespace isai

#endif  // !ISAI_GENEPI_POLY_H_INCLUDED
