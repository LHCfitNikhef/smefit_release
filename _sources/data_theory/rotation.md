```eval_rst
.. _rotation:
```
# Basis rotation
It is possible to perform a fit in any desired basis.
The user then needs to specify the rotation matrix between the fitting and the Warsaw basis.
This has to be specified in a ``.json`` file, whose absolute path has to be placed in the runcard
```yaml
rotation: /path/to/rotatio/matrix/rotation.json
```
The file ``rotation.json`` will look like

```yaml
{
    name: "example",
    warsaw: [
        "Ow_1",
        "Ow_2",
        "Ow_3",
        "Ow_4",
        ...
    ],
    new_basis: [
        "Of_1",
        "Of_2",
        "Of_3",
        "Of_4",
        ...
    ],
    matrix: [
        [
            R_11,
            R_12,
            R_13,
            R_14,
            ...
        ],
        ...
        [
            R_n1,
            R_n2,
            R_n3,
            R_n4,
            ...
        ]
    ]


}

```

``warsaw`` : list of the ``n`` operators ``Ow_i`` in the Warsaw basis entering the theory tables,

``fit_basis`` : list of the ``n`` operators defining the fitting basis ``Of_i``,

``matrix`` : the ``n x n`` rotation matrix ``R`` expressing the new basis in terms of the Warsaw one
``Of_i = R_ij Ow_j``.
