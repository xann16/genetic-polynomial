#pragma once

#ifndef ISAI_GENEPI_POLY_H_INCLUDED
#define ISAI_GENEPI_POLY_H_INCLUDED

#include "prng.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace isai
{

  struct data_point_t
  {
    double x;
    double y;
  };

  using training_data_t = std::vector< data_point_t >;


  template < std::size_t N >
  class polynomial_t
  {
  public:
    polynomial_t() noexcept : m_data()
    {
      std::memset( m_data.data(), 0, size() * sizeof( double ) );
    }

    polynomial_t( polynomial_t const & ) = default;
    polynomial_t( polynomial_t && ) noexcept = default;
    polynomial_t &operator=( polynomial_t const & ) = default;
    polynomial_t &operator=( polynomial_t && ) noexcept = default;
    ~polynomial_t() noexcept = default;

    polynomial_t( std::initializer_list< double > il ) noexcept : m_data()
    {
      assert( il.size() == size() );
      auto index = std::size_t{ 0 };
      for ( auto &&val : il )
      {
        m_data[ index++ ] = val;
      }
    }

    explicit polynomial_t( double *vals_p ) noexcept : m_data()
    {
      std::memcpy( m_data.data(), vals_p, size() * sizeof( double ) );
    }

    double operator[]( std::size_t index ) const noexcept
    {
      assert( index < size() );
      return m_data[ index ];
    }

    double &operator[]( std::size_t index ) noexcept
    {
      assert( index < size() );
      return m_data[ index ];
    }

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

    constexpr std::size_t size() const noexcept { return N + 1; }

    constexpr std::size_t order() const noexcept { return N; }

    training_data_t get_training_data( std::size_t count, double argmin = -10.0,
                                       double argmax = 10.0 )
    {
      auto args = prng_t::get_uniform_doubles( count, argmin, argmax );
      /*
            auto args = std::vector< double >{};
            args.reserve( count );
            for ( auto i = std::size_t{ 0 }; i < count; i++ )
            {
              args.emplace_back(
                argmin + ( ( ( argmax - argmin ) / static_cast< double >( count
         ) ) * static_cast< double >( i ) ) );
            }
      */
      auto res = training_data_t{};
      res.reserve( count );
      std::transform( std::begin( args ), std::end( args ),
                      std::back_inserter( res ), [this]( double arg ) {
                        return data_point_t{ arg, ( *this )( arg ) };
                      } );
      return res;
    }

  private:
    std::array< double, N + 1 > m_data;
  };

}  // namespace isai

#endif  // !ISAI_GENEPI_POLY_H_INCLUDED
